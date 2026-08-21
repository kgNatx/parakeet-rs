use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
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
    /// Encoder subsampling factor. Matches the stride `ParakeetTDTDecoder` uses to turn
    /// `frame_indices` into timestamps; the two must agree or timestamps and frames desynchronize.
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
        // Run encoder
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;

        // Run greedy decoding with decoder_joint
        let (tokens, frame_indices, durations, confidences) =
            self.greedy_decode(&encoder_out, encoder_len)?;

        Ok((tokens, frame_indices, durations, confidences))
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
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;

        let (tokens, frame_indices, durations, confidences) =
            self.greedy_decode(&encoder_out, encoder_len)?;

        Ok((tokens, frame_indices, durations, confidences, encoder_out))
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
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>)> {
        // encoder_out shape: [batch, encoder_dim, time]
        let encoder_dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2];
        let vocab_size = self.config.vocab_size;
        let max_tokens_per_step = 10;
        let blank_id = vocab_size - 1;

        // States: (num_layers=2, batch=1, hidden_dim=640)
        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut tokens = Vec::new();
        let mut frame_indices = Vec::new();
        let mut durations = Vec::new();
        let mut confidences = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut last_emitted_token = blank_id as i32;

        // Frame-by-frame RNN-T/TDT greedy decoding
        while t < time_steps {
            // Get single encoder frame: slice [0, :, t] and reshape to [1, encoder_dim, 1]
            let frame = encoder_out.slice(ndarray::s![0, .., t]).to_owned();
            let frame_reshaped = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            // Current token for prediction network
            let targets = Array2::from_shape_vec((1, 1), vec![last_emitted_token])
                .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

            // Run decoder_joint
            let outputs = self.decoder_joint.run(ort::inputs!(
                "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)?,
                "targets" => ort::value::Value::from_array(targets)?,
                "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
                "input_states_1" => ort::value::Value::from_array(state_h.clone())?,
                "input_states_2" => ort::value::Value::from_array(state_c.clone())?
            ))?;

            // Extract logits
            let (_, logits_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

            // TDT outputs vocab_size + 5 durations
            let vocab_logits: Vec<f32> = logits_data.iter().take(vocab_size).copied().collect();
            let duration_logits: Vec<f32> = logits_data.iter().skip(vocab_size).copied().collect();

            let token_id = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            // Compute softmax probability of the winning token for confidence scoring
            let token_confidence = if token_id != blank_id {
                let max_logit = vocab_logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = vocab_logits.iter().map(|&l| (l - max_logit).exp()).sum();
                (vocab_logits[token_id] - max_logit).exp() / exp_sum
            } else {
                0.0
            };

            let duration_step = if !duration_logits.is_empty() {
                duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            // Check if blank token
            if token_id != blank_id {
                // Update states when we emit a token
                if let Ok((h_shape, h_data)) =
                    outputs["output_states_1"].try_extract_tensor::<f32>()
                {
                    let dims = h_shape.as_ref();
                    state_h = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        h_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;
                }
                if let Ok((c_shape, c_data)) =
                    outputs["output_states_2"].try_extract_tensor::<f32>()
                {
                    let dims = c_shape.as_ref();
                    state_c = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        c_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;
                }

                tokens.push(token_id);
                frame_indices.push(t);
                durations.push(duration_step);
                confidences.push(token_confidence);
                last_emitted_token = token_id as i32;
                emitted_tokens += 1;
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

        Ok((tokens, frame_indices, durations, confidences))
    }
}
