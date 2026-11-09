use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::phrase_boost::{BoostCursor, BoostTrace, PhraseBooster};
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use std::path::{Path, PathBuf};

/// TDT model configs
#[derive(Debug, Clone)]
pub struct TDTModelConfig {
    pub vocab_size: usize,
}

impl TDTModelConfig {
    /// Create config with specified vocab size
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

pub struct ParakeetTDTModel {
    encoder: Session,
    decoder_joint: Session,
    config: TDTModelConfig,
}

/// Encoder hidden states for one utterance, time-major.
///
/// The FastConformer encoder subsamples 8x, so each entry of `frames` covers
/// `8 * hop_length / sample_rate` seconds — 80 ms at the TDT defaults. That is coarse next to the
/// 10 ms MFCC hop it replaces: a 600 ms word is about 7 frames. In exchange each frame is a
/// ~1024-dim representation the model learned for distinguishing what was said, rather than 13
/// cepstral coefficients that also encode who said it and on what microphone.
#[derive(Debug, Clone)]
pub struct EncoderStates {
    /// Encoder width (`frames[t].len()` for every `t`).
    pub dim: usize,
    /// Time-major frames: `frames[t][d]`.
    pub frames: Vec<Vec<f32>>,
    /// Feature-extractor hop, in samples, before encoder subsampling.
    pub hop_length: usize,
    /// Sample rate the hop is measured against.
    pub sample_rate: usize,
}

impl EncoderStates {
    /// Encoder subsampling factor. `decoder_tdt::token_start_seconds` turns `frame_indices` into
    /// timestamps with this same constant, so timestamps and frames cannot desynchronize.
    pub const ENCODER_STRIDE: usize = 8;

    /// Seconds covered by one encoder frame.
    pub fn seconds_per_frame(&self) -> f64 {
        (Self::ENCODER_STRIDE * self.hop_length) as f64 / self.sample_rate as f64
    }

    /// Frames overlapping `[start, end)` seconds, clamped to what the encoder actually produced.
    ///
    /// Returns an empty slice when the range is empty or lies past the end. A range shorter than
    /// one frame still yields the single frame containing `start`, so a very short word is never
    /// silently dropped.
    pub fn slice_seconds(&self, start: f64, end: f64) -> &[Vec<f32>] {
        let spf = self.seconds_per_frame();
        let n = self.frames.len();
        if n == 0 || end <= start {
            return &[];
        }
        let lo = ((start / spf).floor().max(0.0) as usize).min(n.saturating_sub(1));
        let hi = ((end / spf).ceil() as usize).clamp(lo + 1, n);
        &self.frames[lo..hi]
    }

    /// Transpose ONNX `[batch, encoder_dim, time]` into time-major frames.
    pub(crate) fn from_encoder_out(
        encoder_out: &Array3<f32>,
        hop_length: usize,
        sample_rate: usize,
    ) -> Self {
        let dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2];
        let mut frames = Vec::with_capacity(time_steps);
        for t in 0..time_steps {
            frames.push(encoder_out.slice(ndarray::s![0, .., t]).to_vec());
        }
        Self { dim, frames, hop_length, sample_rate }
    }
}

impl ParakeetTDTModel {
    /// Load TDT model from directory containing encoder and decoder_joint ONNX files
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing encoder and decoder_joint ONNX files
    /// * `exec_config` - Execution configuration for ONNX runtime
    /// * `vocab_size` - Vocabulary size (number of tokens including blank)
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find encoder and decoder_joint files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir)?;

        let config = TDTModelConfig::new(vocab_size);

        // Load encoder
        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder_joint
        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_joint_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }
    //file names simply from: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        // fallback
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("encoder") && name.ends_with(".onnx") {
                        return Ok(path);
                    }
                }
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder_joint.onnx",
            "decoder-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No decoder_joint model found in {}",
            dir.display()
        )))
    }

    /// Run greedy decoding - returns (token_ids, frame_indices, durations, confidences)
    pub fn forward(
        &mut self,
        features: Array2<f32>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>)> {
        let (d, _) = self.decode(features, None)?;
        Ok((d.tokens, d.frame_indices, d.durations, d.confidences))
    }

    /// Like [`forward`](Self::forward), but also hands back the encoder hidden states that
    /// greedy decoding just consumed.
    ///
    /// `run_encoder` already computes these on every utterance and `forward` drops them on the
    /// floor. Returning them costs one move, no extra inference. Callers doing acoustic matching
    /// want the encoder representation rather than raw filterbank features, because the encoder is
    /// trained to discard speaker, channel and prosody while keeping phonetic identity.
    ///
    /// The returned array is `[batch, encoder_dim, time]` — the same layout `greedy_decode`
    /// indexes, so `frame_indices` line up with its time axis directly. Note the encoder subsamples
    /// 8x, so one time step is `8 * hop_length / sample_rate` seconds (80 ms at the TDT defaults).
    pub fn forward_with_encoder(
        &mut self,
        features: Array2<f32>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>, Array3<f32>)> {
        let (d, encoder_out) = self.decode(features, None)?;
        Ok((d.tokens, d.frame_indices, d.durations, d.confidences, encoder_out))
    }

    /// Encode `features` and decode greedily, optionally biased by `booster`; returns the decode
    /// and the encoder output it ran on (`[batch, encoder_dim, time]`).
    ///
    /// With `booster = None` this is exactly [`forward_with_encoder`](Self::forward_with_encoder).
    /// The caller must have checked that `booster` was built for this model's vocabulary
    /// (`PhraseBooster::check_vocab`); token ids are all this level sees.
    pub(crate) fn decode(
        &mut self,
        features: Array2<f32>,
        booster: Option<&PhraseBooster>,
    ) -> Result<(Decoded, Array3<f32>)> {
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;
        let decoded = self.greedy_decode(&encoder_out, encoder_len, booster)?;
        Ok((decoded, encoder_out))
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        // TDT encoder expects (batch, features, time) not (batch, time, features)
        let input = features
            .t()
            .to_shape((batch_size, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to reshape encoder input: {e}")))?
            .to_owned();

        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(input)?;
        let length_value = ort::value::Value::from_array(input_length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let encoder_out = &outputs["outputs"];
        let encoder_lens = &outputs["encoded_lengths"];

        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let (_, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder lengths: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape_dims:?}"
            )));
        }

        let b = shape_dims[0] as usize;
        let t = shape_dims[1] as usize;
        let d = shape_dims[2] as usize;

        let encoder_array = Array3::from_shape_vec((b, t, d), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?;

        // TDT encoder outputs [batch, encoder_dim, time] directly
        Ok((encoder_array, lens_data[0]))
    }

    fn greedy_decode(
        &mut self,
        encoder_out: &Array3<f32>,
        _encoder_len: i64,
        booster: Option<&PhraseBooster>,
    ) -> Result<Decoded> {
        // encoder_out shape: [batch, encoder_dim, time]
        let time_steps = encoder_out.shape()[2];
        let vocab_size = self.config.vocab_size;
        let mut joint = OnnxJoint::new(&mut self.decoder_joint, encoder_out);
        decode_utterance(&mut joint, time_steps, vocab_size, booster)
    }
}

/// One step of the TDT prediction network + joint, as the greedy walk sees it.
///
/// The walk calls `logits` for frame `t` given the last emitted token, decides what to do, and
/// calls `accept` only when it emits a non-blank token. `accept` adopts the prediction-network
/// state that the most recent `logits` call produced; if that call produced no usable state, the
/// previous state is kept. Implementations hold their own state, so the walk never sees it.
pub(crate) trait Joint {
    /// Logits for frame `t` with `last_token` fed to the prediction network: the `vocab_size`
    /// token logits (blank last), followed by the duration logits.
    fn logits(&mut self, t: usize, last_token: i32) -> Result<Vec<f32>>;
    /// Adopt the prediction-network state that the most recent `logits` call produced.
    fn accept(&mut self) -> Result<()>;
    /// Return to the state before the first `logits` call, so the utterance can be walked again.
    fn reset(&mut self) -> Result<()>;
}

/// The result of a greedy TDT walk, one entry per emitted (non-blank) token.
pub(crate) struct Decoded {
    pub tokens: Vec<usize>,
    pub frame_indices: Vec<usize>,
    pub durations: Vec<usize>,
    pub confidences: Vec<f32>,
    /// What the booster's cursor saw, one `changed` entry per emitted token; `None` exactly when
    /// the decode ran without a booster (an empty booster still leaves a trace).
    pub trace: Option<BoostTrace>,
}

/// Index of the largest value. Of equal maxima the LAST wins, and NaN compares as Equal
/// (`Iterator::max_by` with `partial_cmp -> Equal`). `None` only for an empty slice.
pub(crate) fn argmax_last(xs: &[f32]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
}

/// Softmax probability of the winning token, used as its confidence; 0.0 for blank.
fn token_confidence(vocab_logits: &[f32], token_id: usize, blank_id: usize) -> f32 {
    if token_id != blank_id {
        let max_logit = vocab_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = vocab_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        (vocab_logits[token_id] - max_logit).exp() / exp_sum
    } else {
        0.0
    }
}

/// The TDT blank token's id: the last of the `vocab_size` token logits. The ONE definition, shared
/// by the decode walk and the phrase booster (which must never bias toward blank).
pub(crate) fn tdt_blank_id(vocab_size: usize) -> usize {
    vocab_size.saturating_sub(1)
}

/// Frame-by-frame TDT greedy decoding over `time_steps` encoder frames.
///
/// Knows the walk: the token and duration argmaxes, that a duration > 0 jumps that many frames,
/// that a blank (or `max_tokens_per_step` symbols on one frame) advances one frame, and that the
/// prediction network advances only on emission. It knows nothing about ONNX; `joint` does.
///
/// With a booster's `cursor`, the cursor picks each token (blank vs non-blank stays the unbiased
/// decision) and is told every emitted token; nothing else in the walk changes. Confidence is
/// always the softmax of the chosen token under the UNBIASED logits. `cursor = None` is today's
/// decode. One walk only: [`decode_utterance`] owns when to walk again.
pub(crate) fn decode_steps(
    joint: &mut dyn Joint,
    time_steps: usize,
    vocab_size: usize,
    cursor: Option<BoostCursor<'_>>,
) -> Result<Decoded> {
    let max_tokens_per_step = 10;
    let blank_id = tdt_blank_id(vocab_size);

    let mut tokens = Vec::new();
    let mut frame_indices = Vec::new();
    let mut durations = Vec::new();
    let mut confidences = Vec::new();

    let mut t = 0;
    let mut emitted_tokens = 0;
    let mut last_emitted_token = blank_id as i32;
    let mut cursor = cursor;

    // Frame-by-frame RNN-T/TDT greedy decoding
    while t < time_steps {
        let logits_data = joint.logits(t, last_emitted_token)?;

        // TDT outputs vocab_size + 5 durations
        let vocab_logits: Vec<f32> = logits_data.iter().take(vocab_size).copied().collect();
        let duration_logits: Vec<f32> = logits_data.iter().skip(vocab_size).copied().collect();

        let token_id = match cursor.as_mut() {
            Some(c) => c.pick(&vocab_logits),
            None => argmax_last(&vocab_logits).unwrap_or(blank_id),
        };

        // Softmax probability of the chosen token under the UNBIASED logits, for confidence
        let token_confidence = token_confidence(&vocab_logits, token_id, blank_id);

        let duration_step = if !duration_logits.is_empty() {
            argmax_last(&duration_logits).unwrap_or(0)
        } else {
            0
        };

        // Check if blank token
        if token_id != blank_id {
            // Update states when we emit a token
            joint.accept()?;

            tokens.push(token_id);
            frame_indices.push(t);
            durations.push(duration_step);
            confidences.push(token_confidence);
            last_emitted_token = token_id as i32;
            emitted_tokens += 1;
            if let Some(c) = cursor.as_mut() {
                c.emitted(token_id);
            }
        }
        // When duration > 0, skip frames according to duration prediction
        // Otherwise advance by 1 on blank or when max tokens reached
        if duration_step > 0 {
            t += duration_step;
            emitted_tokens = 0;
        } else if token_id == blank_id || emitted_tokens >= max_tokens_per_step {
            t += 1;
            emitted_tokens = 0;
        }
    }

    Ok(Decoded {
        tokens,
        frame_indices,
        durations,
        confidences,
        trace: cursor.map(|c| c.finish()),
    })
}

/// Greedy-decode one utterance, boosted toward `booster`'s phrases when given.
///
/// **Boosting never decides how an utterance ends.** If boosting changed any token of the
/// decode's last word ([`PhraseBooster::forced_last_word`]), the utterance is walked again from
/// the start with boosting stopped at that word's first token: the result is the boosted decode up
/// to that word and the model's own decode from there. The second walk happens only then, and its
/// prefix is the first walk's exactly (same logits, same picks).
///
/// Why: the end of the audio is where speech is most often cut off, and with nothing after it to
/// disagree, the booster finishes a fragment as a custom word ("Mary B." -> "Mary Bowerbird").
/// Measured 2026-11-09 (Speakey `scoreboard hits`): on Earnings-21, 161 of the 249 forced phrases
/// that the reference did not contain were the decode's last word; of the forced phrases it did
/// contain, 0 of 45 (Earnings-21) and 1 of 241 (LibriSpeech test-clean) were.
///
/// Known limit, accepted: one second walk only. If the model's own tail adds no word, the last
/// word is an earlier one, which boosting may have changed.
pub(crate) fn decode_utterance(
    joint: &mut dyn Joint,
    time_steps: usize,
    vocab_size: usize,
    booster: Option<&PhraseBooster>,
) -> Result<Decoded> {
    let first = decode_steps(joint, time_steps, vocab_size, booster.map(PhraseBooster::cursor))?;
    let (Some(b), Some(trace)) = (booster, first.trace.as_ref()) else {
        return Ok(first);
    };
    let Some(word) = b.forced_last_word(&first.tokens, trace) else {
        return Ok(first);
    };
    joint.reset()?;
    decode_steps(joint, time_steps, vocab_size, Some(b.cursor_until(word)))
}

/// Raw `(dims, data)` of one extracted output state, not yet shaped.
type RawState = (Vec<i64>, Vec<f32>);

/// The ONNX `decoder_joint` session as a [`Joint`].
///
/// Prediction-network states start at zero `(num_layers=2, batch=1, hidden_dim=640)`. Each
/// `logits` call first clears `pending`, then stashes every output state it could extract;
/// `accept` shapes those into the live states. A state whose extraction failed stays as it was,
/// and a shape error surfaces only on emission: exactly the pre-seam `if let Ok` behaviour.
struct OnnxJoint<'a> {
    session: &'a mut Session,
    encoder_out: &'a Array3<f32>,
    state_h: Array3<f32>,
    state_c: Array3<f32>,
    /// `[output_states_1, output_states_2]` from the most recent `logits` call.
    pending: Option<[Option<RawState>; 2]>,
}

impl<'a> OnnxJoint<'a> {
    fn new(session: &'a mut Session, encoder_out: &'a Array3<f32>) -> Self {
        Self {
            session,
            encoder_out,
            state_h: Array3::<f32>::zeros((2, 1, 640)),
            state_c: Array3::<f32>::zeros((2, 1, 640)),
            pending: None,
        }
    }
}

impl Joint for OnnxJoint<'_> {
    fn logits(&mut self, t: usize, last_token: i32) -> Result<Vec<f32>> {
        self.pending = None;
        let encoder_dim = self.encoder_out.shape()[1];

        // Get single encoder frame: slice [0, :, t] and reshape to [1, encoder_dim, 1]
        let frame = self.encoder_out.slice(ndarray::s![0, .., t]).to_owned();
        let frame_reshaped = frame
            .to_shape((1, encoder_dim, 1))
            .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
            .to_owned();

        // Current token for prediction network
        let targets = Array2::from_shape_vec((1, 1), vec![last_token])
            .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

        // Run decoder_joint
        let outputs = self.session.run(ort::inputs!(
            "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)?,
            "targets" => ort::value::Value::from_array(targets)?,
            "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
            "input_states_1" => ort::value::Value::from_array(self.state_h.clone())?,
            "input_states_2" => ort::value::Value::from_array(self.state_c.clone())?
        ))?;

        // Extract logits
        let (_, logits_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
        let logits = logits_data.to_vec();

        let raw = |name: &str| -> Option<RawState> {
            outputs[name]
                .try_extract_tensor::<f32>()
                .ok()
                .map(|(shape, data)| (shape.as_ref().to_vec(), data.to_vec()))
        };
        self.pending = Some([raw("output_states_1"), raw("output_states_2")]);

        Ok(logits)
    }

    fn accept(&mut self) -> Result<()> {
        let Some([h, c]) = self.pending.take() else {
            return Ok(());
        };
        if let Some((dims, h_data)) = h {
            self.state_h = Array3::from_shape_vec(
                (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                h_data,
            )
            .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;
        }
        if let Some((dims, c_data)) = c {
            self.state_c = Array3::from_shape_vec(
                (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                c_data,
            )
            .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.pending = None;
        Ok(())
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    /// Vocab of 4 tokens, blank last (id 3), then 5 duration logits.
    const VOCAB: usize = 4;
    const BLANK: usize = 3;
    const N_DUR: usize = 5;

    #[test]
    fn the_blank_is_the_last_token() {
        assert_eq!(tdt_blank_id(VOCAB), BLANK);
    }

    /// Logits whose token argmax is `tok` and duration argmax is `dur`.
    fn step(tok: usize, dur: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; VOCAB + N_DUR];
        v[tok] = 5.0;
        v[VOCAB + dur] = 5.0;
        v
    }

    /// A `Joint` that replays scripted logits. A "state" is named by the index of the call that
    /// produced it; a call whose extraction is scripted to fail produces no state, so accepting
    /// after it must leave the previous state in place.
    struct ScriptedJoint {
        script: Vec<(Vec<f32>, bool)>,
        next: usize,
        calls: Vec<(usize, i32)>,
        pending: Option<usize>,
        state: Option<usize>,
        accepted: Vec<usize>,
    }

    impl ScriptedJoint {
        fn new(script: Vec<(Vec<f32>, bool)>) -> Self {
            Self { script, next: 0, calls: Vec::new(), pending: None, state: None, accepted: Vec::new() }
        }
        fn ok(steps: Vec<Vec<f32>>) -> Self {
            Self::new(steps.into_iter().map(|s| (s, true)).collect())
        }
        fn times(&self) -> Vec<usize> {
            self.calls.iter().map(|&(t, _)| t).collect()
        }
        fn last_tokens(&self) -> Vec<i32> {
            self.calls.iter().map(|&(_, l)| l).collect()
        }
    }

    impl Joint for ScriptedJoint {
        fn logits(&mut self, t: usize, last_token: i32) -> Result<Vec<f32>> {
            self.pending = None;
            let i = self.next;
            assert!(i < self.script.len(), "script exhausted at call {i} (t={t})");
            self.next += 1;
            self.calls.push((t, last_token));
            let (logits, extract_ok) = self.script[i].clone();
            if extract_ok {
                self.pending = Some(i);
            }
            Ok(logits)
        }
        fn accept(&mut self) -> Result<()> {
            if let Some(s) = self.pending {
                self.state = Some(s);
                self.accepted.push(s);
            }
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            unreachable!("the walk tests decode one walk")
        }
    }

    #[test]
    fn blank_advances_one_frame() {
        let mut j = ScriptedJoint::ok(vec![step(BLANK, 0), step(BLANK, 0), step(BLANK, 0)]);
        let d = decode_steps(&mut j, 3, VOCAB, None).unwrap();
        assert_eq!(j.times(), vec![0, 1, 2]);
        assert!(d.tokens.is_empty());
        assert!(j.accepted.is_empty());
    }

    #[test]
    fn duration_skips_frames_and_resets_the_symbol_count() {
        // 9 symbols on frame 0, then a 10th with duration 2: jump to t=2 and reset the count.
        // If the count were not reset, the first symbol on t=2 would force an advance to t=3.
        let mut steps = vec![step(1, 0); 9];
        steps.push(step(2, 2));
        steps.push(step(1, 0));
        steps.push(step(BLANK, 0));
        let mut j = ScriptedJoint::ok(steps);
        let d = decode_steps(&mut j, 3, VOCAB, None).unwrap();
        let mut want = vec![0; 10];
        want.extend([2, 2]);
        assert_eq!(j.times(), want);
        assert_eq!(d.tokens.len(), 11);
        assert_eq!(d.durations[9], 2);
        assert_eq!(d.frame_indices[9], 0);
        assert_eq!(d.frame_indices[10], 2);
    }

    #[test]
    fn duration_on_the_first_frame_jumps_straight_there() {
        // tok1 with duration argmax 2 at t=0 → next call at t=2.
        let mut j = ScriptedJoint::ok(vec![step(1, 2), step(BLANK, 0)]);
        let d = decode_steps(&mut j, 3, VOCAB, None).unwrap();
        assert_eq!(j.times(), vec![0, 2]);
        assert_eq!(d.tokens, vec![1]);
        assert_eq!(d.frame_indices, vec![0]);
        assert_eq!(d.durations, vec![2]);
    }

    #[test]
    fn ten_symbols_on_one_frame_force_an_advance() {
        let mut steps = vec![step(1, 0); 10];
        steps.push(step(BLANK, 0));
        let mut j = ScriptedJoint::ok(steps);
        let d = decode_steps(&mut j, 2, VOCAB, None).unwrap();
        let mut want = vec![0; 10];
        want.push(1);
        assert_eq!(j.times(), want);
        assert_eq!(d.tokens, vec![1; 10]);
    }

    #[test]
    fn state_is_accepted_only_when_a_token_is_emitted() {
        // [tok1, blank, tok2]: two accepts; the prediction network sees blank, then tok1 twice
        // (a blank does not change the last emitted token).
        let mut j = ScriptedJoint::ok(vec![step(1, 0), step(BLANK, 0), step(2, 1)]);
        let d = decode_steps(&mut j, 2, VOCAB, None).unwrap();
        assert_eq!(j.last_tokens(), vec![BLANK as i32, 1, 1]);
        assert_eq!(j.accepted, vec![0, 2]);
        assert_eq!(d.tokens, vec![1, 2]);
        assert_eq!(d.frame_indices, vec![0, 1]);
    }

    #[test]
    fn a_failed_state_extraction_keeps_the_previous_state() {
        // Step 0 emits with a good state; step 1 emits but its state extraction fails, so the
        // state after step 1 must still be step 0's. The token is still emitted and fed back.
        let mut j = ScriptedJoint::new(vec![
            (step(1, 0), true),
            (step(2, 0), false),
            (step(BLANK, 0), true),
        ]);
        let d = decode_steps(&mut j, 1, VOCAB, None).unwrap();
        assert_eq!(d.tokens, vec![1, 2]);
        assert_eq!(j.state, Some(0));
        assert_eq!(j.accepted, vec![0]);
        assert_eq!(j.last_tokens(), vec![BLANK as i32, 1, 2]);
    }

    #[test]
    fn equal_maxima_pick_the_last_index() {
        assert_eq!(argmax_last(&[1.0, 5.0, 5.0, 0.0]), Some(2));
        assert_eq!(argmax_last(&[]), None);
        // NaN compares as Equal, so it is "replaced" by whatever follows it.
        assert_eq!(argmax_last(&[5.0, f32::NAN, 1.0]), Some(2));
    }

    #[test]
    fn confidence_is_the_softmax_of_the_winner_and_zero_for_blank() {
        let mut first = vec![2.0, 1.0, 0.0, -1.0];
        first.extend([0.0, 5.0, 0.0, 0.0, 0.0]); // duration 1
        let mut j = ScriptedJoint::ok(vec![first, step(BLANK, 0)]);
        let d = decode_steps(&mut j, 2, VOCAB, None).unwrap();
        let sum: f32 = [2.0f32, 1.0, 0.0, -1.0].iter().map(|l| (l - 2.0).exp()).sum();
        assert_eq!(d.tokens, vec![0]);
        assert!((d.confidences[0] - 1.0 / sum).abs() < 1e-6);
        // A blank is never recorded; its confidence, were it asked for, is 0.
        assert_eq!(d.confidences.len(), 1);
        assert_eq!(token_confidence(&[0.0, 0.0, 0.0, 9.0], BLANK, BLANK), 0.0);
    }
}

#[cfg(test)]
mod boost_decode_tests {
    use super::*;
    use crate::phrase_boost::{BoostPhrase, PhraseBooster, RawHit};

    /// A 10-piece vocab, blank last. `cairn` = `▁ca irn`, `dogs` = `▁dog s`.
    const V: usize = 10;
    const BLANK: usize = 9;
    const N_DUR: usize = 5;
    const CA: usize = 1;
    const IRN: usize = 2;
    const THE: usize = 3;
    const DOG: usize = 4;

    fn pieces() -> Vec<String> {
        ["<unk>", "▁ca", "irn", "▁the", "▁dog", "s", "▁cat", "▁a", "b", "<blk>"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn booster(phrases: &[&str], strength: f32) -> PhraseBooster {
        let p: Vec<BoostPhrase> = phrases.iter().map(|s| BoostPhrase::new(*s, strength)).collect();
        let v = pieces();
        let t = crate::phrase_boost::fixture_tokenizer(&v, BLANK);
        PhraseBooster::new(&v, BLANK, t, &p).unwrap()
    }

    /// Replays logits by call index, whatever token is fed back, and records the calls.
    struct Replay {
        script: Vec<Vec<f32>>,
        calls: Vec<(usize, i32)>,
        /// Call indices whose state was accepted, i.e. that emitted a token.
        accepted: Vec<usize>,
        /// `calls.len()` when the current walk began; `reset` starts a new walk.
        walk_start: usize,
        /// How many times `reset` was called.
        resets: usize,
    }

    impl Replay {
        fn new(script: Vec<Vec<f32>>) -> Self {
            Self { script, calls: Vec::new(), accepted: Vec::new(), walk_start: 0, resets: 0 }
        }
    }

    impl Joint for Replay {
        fn logits(&mut self, t: usize, last_token: i32) -> Result<Vec<f32>> {
            let i = self.calls.len() - self.walk_start;
            assert!(i < self.script.len(), "script exhausted at call {i} (t={t})");
            self.calls.push((t, last_token));
            Ok(self.script[i].clone())
        }
        fn accept(&mut self) -> Result<()> {
            self.accepted.push(self.calls.len() - 1);
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.walk_start = self.calls.len();
            self.resets += 1;
            Ok(())
        }
    }

    /// 300 steps from a fixed-seed LCG. Token logits are small integers, so exact ties are
    /// common (blank included); the duration head picks 0..4, so frames are skipped.
    fn lcg_script() -> Vec<Vec<f32>> {
        let mut s: u64 = 0x5eed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        (0..300)
            .map(|_| {
                let mut v: Vec<f32> = (0..V).map(|_| (next() % 4) as f32).collect();
                // Duration 0 most often, so several symbols land on one frame.
                let dur = match next() % 8 {
                    0..=3 => 0,
                    d => (d - 3) as usize,
                };
                let mut durs = vec![0.0f32; N_DUR];
                durs[dur] = 1.0;
                v.extend(durs);
                v
            })
            .collect()
    }

    /// Encoder frames for the LCG script: few enough that 300 calls always suffice.
    const LCG_FRAMES: usize = 40;

    fn run(script: &[Vec<f32>], frames: usize, b: Option<&PhraseBooster>) -> (Decoded, Replay) {
        let mut j = Replay::new(script.to_vec());
        let d = decode_steps(&mut j, frames, V, b.map(PhraseBooster::cursor)).unwrap();
        (d, j)
    }

    /// Like `run`, through [`decode_utterance`] (the end rule included).
    fn utterance(script: &[Vec<f32>], frames: usize, b: Option<&PhraseBooster>) -> (Decoded, Replay) {
        let mut j = Replay::new(script.to_vec());
        let d = decode_utterance(&mut j, frames, V, b).unwrap();
        (d, j)
    }

    fn assert_identical(a: &Decoded, b: &Decoded) {
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.frame_indices, b.frame_indices);
        assert_eq!(a.durations, b.durations);
        let bits = |d: &Decoded| d.confidences.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a), bits(b));
    }

    #[test]
    fn the_lcg_script_exercises_blanks_ties_and_skips() {
        let script = lcg_script();
        let (d, j) = run(&script, LCG_FRAMES, None);
        let calls = j.calls.len();
        let vocab = |i: usize| &script[i][..V];
        let max = |i: usize| vocab(i).iter().cloned().fold(f32::MIN, f32::max);
        assert!((0..calls).any(|i| argmax_last(vocab(i)) == Some(BLANK)), "a blank");
        assert!(
            (0..calls).any(|i| vocab(i).iter().filter(|&&x| x == max(i)).count() > 1),
            "a tie"
        );
        assert!(d.durations.iter().any(|&x| x > 1), "a skip");
        assert!(!d.tokens.is_empty());
    }

    #[test]
    fn no_booster_and_an_empty_booster_decode_identically() {
        let script = lcg_script();
        let (plain, pj) = run(&script, LCG_FRAMES, None);
        assert!(plain.trace.is_none());
        let empty = booster(&[], 5.0);
        assert!(empty.is_empty());
        let (boosted, bj) = run(&script, LCG_FRAMES, Some(&empty));
        assert_identical(&plain, &boosted);
        assert_eq!(pj.calls, bj.calls);
        // The empty booster really ran its cursor: one entry per emitted token, none changed.
        let trace = boosted.trace.expect("a booster always leaves a trace");
        assert_eq!(trace.changed.len(), boosted.tokens.len());
        assert!(trace.changed.iter().all(|&c| !c));
        assert!(trace.hits.is_empty());
    }

    #[test]
    fn zero_strength_decodes_like_no_booster() {
        let script = lcg_script();
        let (plain, pj) = run(&script, LCG_FRAMES, None);
        // A phrase at strength 0 is skipped, so this is the empty booster's path.
        let b = booster(&["cairn", "dogs"], 0.0);
        assert!(b.is_empty());
        assert_eq!(b.skipped().len(), 2);
        let (boosted, bj) = run(&script, LCG_FRAMES, Some(&b));
        assert_identical(&plain, &boosted);
        assert_eq!(pj.calls, bj.calls);
        assert!(boosted.trace.unwrap().changed.iter().all(|&c| !c));
    }

    #[test]
    fn a_blank_decision_is_never_changed_by_boosting() {
        let script = lcg_script();
        let b = booster(&["cairn", "dogs"], PhraseBooster::MAX_STRENGTH);
        let (boosted, j) = run(&script, LCG_FRAMES, Some(&b));
        // A call emits exactly when its UNBIASED argmax is non-blank.
        let want: Vec<usize> = (0..j.calls.len())
            .filter(|&i| argmax_last(&script[i][..V]) != Some(BLANK))
            .collect();
        assert_eq!(j.accepted, want);
        let trace = boosted.trace.unwrap();
        // Boosting did act on this script, so the identity above is not vacuous.
        assert!(trace.changed.iter().any(|&c| c), "boosting changed nothing");
    }

    /// Logits with token `tok_logits` (others 0) and duration argmax `dur`.
    fn step(tok_logits: &[(usize, f32)], dur: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; V + N_DUR];
        for &(t, x) in tok_logits {
            v[t] = x;
        }
        v[V + dur] = 5.0;
        v
    }

    /// `▁ca` (skipping 2 frames), a blank, then `▁the` narrowly ahead of `irn`.
    fn cairn_script() -> Vec<Vec<f32>> {
        vec![
            step(&[(CA, 3.0)], 2),
            step(&[(BLANK, 3.0)], 0),
            step(&[(THE, 2.0), (IRN, 1.0)], 1),
            step(&[(BLANK, 3.0)], 0),
        ]
    }

    #[test]
    fn boosting_completes_a_phrase_across_blanks_and_skips() {
        let script = cairn_script();
        let (plain, _) = run(&script, 5, None);
        assert_eq!(plain.tokens, vec![CA, THE]);

        let b = booster(&["cairn"], 1.0);
        let (boosted, j) = run(&script, 5, Some(&b));
        assert_eq!(boosted.tokens, vec![CA, IRN]);
        assert_eq!(boosted.frame_indices, vec![0, 3]);
        // The duration head is untouched by boosting.
        assert_eq!(boosted.durations, plain.durations);
        // The prediction network is fed the boosted token.
        assert_eq!(j.calls, vec![(0, BLANK as i32), (2, CA as i32), (3, CA as i32), (4, IRN as i32)]);
        let trace = boosted.trace.unwrap();
        assert_eq!(trace.changed, vec![false, true]);
        assert_eq!(trace.hits, vec![RawHit { phrase: 0, first: 0, last: 1 }]);
    }

    #[test]
    fn confidence_comes_from_the_unbiased_logits() {
        let script = cairn_script();
        let b = booster(&["cairn"], 1.0);
        let (boosted, _) = run(&script, 5, Some(&b));
        assert_eq!(boosted.tokens[1], IRN);
        // Softmax over the raw logits, where `▁the` (2.0) still beats `irn` (1.0).
        let raw = &script[2][..V];
        let sum: f32 = raw.iter().map(|&l| (l - 2.0).exp()).sum();
        let want = (1.0f32 - 2.0).exp() / sum;
        assert_eq!(boosted.confidences[1].to_bits(), want.to_bits());
        assert!(boosted.confidences[1] < 0.5);
    }

    #[test]
    fn boosting_never_decides_the_last_word() {
        // The walk alone boosts `▁the` into `irn`, so "cairn" ends the utterance ...
        let script = cairn_script();
        let b = booster(&["cairn"], 1.0);
        let (walked, _) = run(&script, 5, Some(&b));
        assert_eq!(walked.tokens, vec![CA, IRN]);
        // ... so the utterance is walked again, unboosted from that word's first token on.
        let (plain, _) = run(&script, 5, None);
        let (d, j) = utterance(&script, 5, Some(&b));
        assert_identical(&d, &plain);
        assert_eq!(j.resets, 1);
        let trace = d.trace.unwrap();
        assert!(trace.changed.iter().all(|&c| !c));
        assert!(trace.hits.is_empty());
    }

    #[test]
    fn a_forced_word_followed_by_another_word_stands_in_one_walk() {
        // As `cairn_script`, then `▁dog` the model chooses on its own.
        let mut script = cairn_script();
        script.insert(3, step(&[(DOG, 3.0)], 1));
        let b = booster(&["cairn"], 1.0);
        let (d, j) = utterance(&script, 5, Some(&b));
        assert_eq!(d.tokens, vec![CA, IRN, DOG]);
        assert_eq!(j.resets, 0, "no second walk");
        let (walked, _) = run(&script, 5, Some(&b));
        assert_identical(&d, &walked);
    }

    #[test]
    fn without_a_booster_or_with_nothing_forced_there_is_one_walk() {
        let script = lcg_script();
        let (plain, _) = run(&script, LCG_FRAMES, None);
        let (d, j) = utterance(&script, LCG_FRAMES, None);
        assert_identical(&d, &plain);
        assert_eq!(j.resets, 0);
        let empty = booster(&[], 1.0);
        let (d, j) = utterance(&script, LCG_FRAMES, Some(&empty));
        assert_identical(&d, &plain);
        assert_eq!(j.resets, 0);
    }
}
