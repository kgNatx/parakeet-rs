use crate::audio::{self, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_tdt::{token_start_seconds, ParakeetTDTDecoder};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::{Decoded, EncoderStates, ParakeetTDTModel};
use crate::phrase_boost::{BoostHit, BoostOutcome, PhraseBooster};
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::transcriber::Transcriber;
use crate::vocab::Vocabulary;
use std::path::{Path, PathBuf};

/// Parakeet TDT model for multilingual ASR
pub struct ParakeetTDT {
    model: ParakeetTDTModel,
    decoder: ParakeetTDTDecoder,
    preprocessor_config: PreprocessorConfig,
    feature_cache: FeatureCache,
    model_dir: PathBuf,
}

impl ParakeetTDT {
    /// Load Parakeet TDT model from path with optional configuration.
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "TDT model path must be a directory: {}",
                path.display()
            )));
        }

        let vocab_path = path.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(Error::Config(format!(
                "vocab.txt not found in {}",
                path.display()
            )));
        }

        // TDT-specific preprocessor config (128 features instead of 80)
        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        let exec_config = config.unwrap_or_default();

        // Load vocab first to get the actual vocabulary size
        let vocab = Vocabulary::from_file(&vocab_path)?;
        let vocab_size = vocab.size();

        let model = ParakeetTDTModel::from_pretrained(path, exec_config, vocab_size)?;
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let feature_cache = FeatureCache::from_config(&preprocessor_config);

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            feature_cache,
            model_dir: path.to_path_buf(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }
}

impl ParakeetTDT {
    /// Transcribe, and also return the encoder hidden states the decoder ran on.
    ///
    /// Identical to [`Transcriber::transcribe_samples`] in every observable way except that it
    /// hands back [`EncoderStates`] alongside the transcript. The encoder pass happens either way;
    /// this just stops throwing the result away. Intended for acoustic matching, where the encoder
    /// representation is a far better identity feature than raw MFCCs.
    ///
    /// The transcript's word timestamps and the encoder time axis share one clock, so a word
    /// spanning `[start, end)` seconds maps to encoder frames
    /// `[start / spf, end / spf)` where `spf` is [`EncoderStates::seconds_per_frame`].
    pub fn transcribe_samples_with_encoder(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<(TranscriptionResult, EncoderStates)> {
        let (result, states, _) = self.run(audio, sample_rate, channels, mode, true, None)?;
        // `want_encoder = true` always populates it.
        Ok((result, states.expect("encoder states requested")))
    }

    /// Compile `phrases` into a booster for this model's vocabulary at `strength` (clamped to
    /// `[0, PhraseBooster::MAX_STRENGTH]`), spelling each phrase with the model's canonical
    /// `tokenizer.json` at `tokenizer` (NVIDIA publishes it beside the model). Build once,
    /// reuse across calls. A tokenizer for another vocabulary is an `Error::Config`.
    /// Phrases it cannot boost are listed in [`PhraseBooster::skipped`] with their reason.
    pub fn phrase_booster(
        &self,
        phrases: &[String],
        strength: f32,
        tokenizer: &Path,
    ) -> Result<PhraseBooster> {
        PhraseBooster::from_vocabulary(self.decoder.vocab(), tokenizer, phrases, strength)
    }

    /// [`transcribe_samples_with_encoder`](Self::transcribe_samples_with_encoder), with the greedy
    /// decode biased toward `booster`'s phrases, plus what boosting did.
    ///
    /// `booster = None` is exactly `transcribe_samples_with_encoder`, and its outcome still
    /// carries every emitted token id. An empty booster runs the boosted path and decodes
    /// identically to `None`. A booster built for another model's vocabulary is an
    /// `Error::Config`, never a silent unboosted decode. Hit times share the word timestamps' clock.
    pub fn transcribe_boosted(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
        booster: Option<&PhraseBooster>,
    ) -> Result<(TranscriptionResult, EncoderStates, BoostOutcome)> {
        if let Some(b) = booster {
            b.check_vocab(&self.decoder.vocab().id_to_token)?;
        }
        let (result, states, decoded) =
            self.run(audio, sample_rate, channels, mode, true, booster)?;
        let states = states.expect("encoder states requested");
        let outcome = boost_outcome(
            &decoded,
            booster,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
            states.seconds_per_frame() as f32,
        );
        Ok((result, states, outcome))
    }

    /// Shared body of the transcribe paths. `want_encoder` decides only whether the encoder
    /// output is copied out; the decode is bit-identical either way. `booster` must already have
    /// passed `check_vocab` against this model.
    fn run(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
        want_encoder: bool,
        booster: Option<&PhraseBooster>,
    ) -> Result<(TranscriptionResult, Option<EncoderStates>, Decoded)> {
        // Keep upstream's cached feature extraction (v0.3.6 perf win) AND our
        // confidence return value from forward().
        let features = audio::extract_features_with_cache(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            &self.feature_cache,
        )?;
        let (decoded, enc) = self.model.decode(features, booster)?;
        let encoder = want_encoder.then(|| {
            EncoderStates::from_encoder_out(
                &enc,
                self.preprocessor_config.hop_length,
                self.preprocessor_config.sampling_rate,
            )
        });

        let mut result = self.decoder.decode_with_timestamps(
            &decoded.tokens,
            &decoded.frame_indices,
            &decoded.durations,
            &decoded.confidences,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )?;

        // Apply timestamp mode conversion
        let mode = mode.unwrap_or(TimestampMode::Tokens);
        result.tokens = process_timestamps(&result.tokens, mode);

        // Rebuild full text from processed tokens
        result.text = if mode == TimestampMode::Tokens {
            result
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<String>()
                .trim()
                .to_string()
        } else if mode == TimestampMode::Words {
            let mut out = String::new();
            for (i, word) in result.tokens.iter().map(|t| t.text.as_str()).enumerate() {
                let is_standalone_punct = word.len() == 1
                    && word
                        .chars()
                        .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                if i > 0 && !is_standalone_punct {
                    out.push(' ');
                }
                out.push_str(word);
            }
            out
        } else {
            result
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        Ok((result, encoder, decoded))
    }
}

/// What boosting did to `decoded`, in seconds on the word timestamps' clock. A hit spans from
/// the start of its first token to the end of the frame its last token sits on. `token_ids` is
/// filled whether or not a booster ran.
fn boost_outcome(
    decoded: &Decoded,
    booster: Option<&PhraseBooster>,
    hop_length: usize,
    sample_rate: usize,
    seconds_per_frame: f32,
) -> BoostOutcome {
    let mut outcome = BoostOutcome {
        token_ids: decoded.tokens.clone(),
        ..BoostOutcome::default()
    };
    let (Some(b), Some(trace)) = (booster, decoded.trace.as_ref()) else {
        return outcome;
    };
    let at = |i: usize| token_start_seconds(decoded.frame_indices[i], hop_length, sample_rate);
    outcome.changed_tokens = trace.changed.iter().filter(|&&c| c).count();
    outcome.hits = trace
        .hits
        .iter()
        .map(|h| BoostHit {
            phrase: b.phrases()[h.phrase as usize].clone(),
            start: at(h.first),
            end: at(h.last) + seconds_per_frame,
            changed: trace.changed[h.first..=h.last].iter().any(|&c| c),
        })
        .collect();
    outcome
}

impl Transcriber for ParakeetTDT {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let (result, _, _) = self.run(audio, sample_rate, channels, mode, false, None)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder_tdt::token_start_seconds;
    use crate::model_tdt::Decoded;
    use crate::phrase_boost::{BoostTrace, RawHit};

    fn booster() -> PhraseBooster {
        let v: Vec<String> = ["<unk>", "▁ca", "irn", "▁dog", "s", "▁the", "<blk>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let p: Vec<String> = ["dogs", "cairn"].iter().map(|s| s.to_string()).collect();
        let t = crate::phrase_boost::fixture_tokenizer(&v, 6);
        PhraseBooster::new(&v, 6, t, &p, 1.0).unwrap()
    }

    fn decoded(trace: Option<BoostTrace>) -> Decoded {
        Decoded {
            tokens: vec![5, 1, 2, 3, 4],
            frame_indices: vec![0, 4, 6, 10, 12],
            durations: vec![1, 2, 1, 2, 1],
            confidences: vec![0.9; 5],
            trace,
        }
    }

    #[test]
    fn hits_are_timed_on_the_decoders_clock() {
        let b = booster();
        let trace = BoostTrace {
            changed: vec![false, false, true, false, false],
            hits: vec![
                RawHit { phrase: 1, first: 1, last: 2 },
                RawHit { phrase: 0, first: 3, last: 4 },
            ],
        };
        let spf = 0.08f32;
        let o = boost_outcome(&decoded(Some(trace)), Some(&b), 160, 16000, spf);
        assert_eq!(o.token_ids, vec![5, 1, 2, 3, 4]);
        assert_eq!(o.changed_tokens, 1);
        assert_eq!(o.hits.len(), 2);
        assert_eq!(o.hits[0].phrase, "cairn");
        assert_eq!(o.hits[0].start, token_start_seconds(4, 160, 16000));
        assert_eq!(o.hits[0].end, token_start_seconds(6, 160, 16000) + spf);
        assert!(o.hits[0].changed);
        assert_eq!(o.hits[1].phrase, "dogs");
        assert_eq!(o.hits[1].start, token_start_seconds(10, 160, 16000));
        assert_eq!(o.hits[1].end, token_start_seconds(12, 160, 16000) + spf);
        assert!(!o.hits[1].changed);
    }

    #[test]
    fn an_unboosted_decode_still_reports_its_token_ids() {
        let o = boost_outcome(&decoded(None), None, 160, 16000, 0.08);
        assert_eq!(o.token_ids, vec![5, 1, 2, 3, 4]);
        assert!(o.hits.is_empty());
        assert_eq!(o.changed_tokens, 0);
    }
}
