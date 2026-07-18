use crate::audio::{self, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_tdt::ParakeetTDTDecoder;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::ParakeetTDTModel;
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

    /// Bias the decoder toward the given phrases (e.g. the user's dictionary).
    /// Words are greedy-tokenized over the model's vocab pieces and stored as a
    /// prefix trie. `strength` is the additive logit boost per on-path piece.
    /// Empty `words` (or strength 0) disables biasing.
    pub fn set_bias_phrases(&mut self, words: &[String], strength: f32) {
        use std::collections::HashMap;
        if words.is_empty() || strength == 0.0 {
            self.model.set_bias(None, 0.0);
            return;
        }
        // Build piece -> id map from the vocab, skipping special tokens
        // (anything wrapped in <...> or <|...|>).
        let vocab = self.decoder.vocab(); // from Step 1 (getter or direct field)
        let mut piece_to_id: HashMap<String, u32> = HashMap::new();
        for (id, piece) in vocab.id_to_token.iter().enumerate() {
            if piece.is_empty() || piece.starts_with('<') {
                continue;
            }
            piece_to_id.entry(piece.clone()).or_insert(id as u32);
        }
        // Register BOTH the as-typed and a lowercased tokenization of each word.
        // v3 emits cased text but may lowercase a proper noun mid-sentence; without
        // the lowercase path the trie would silently no-op and produce a FALSE
        // NEGATIVE ("boosting doesn't work") that misjudges the whole approach.
        let mut phrases: Vec<Vec<u32>> = Vec::new();
        for w in words {
            for variant in [w.clone(), w.to_lowercase()] {
                let p = crate::biasing::greedy_tokenize(&variant, &piece_to_id);
                if !p.is_empty() && !phrases.contains(&p) {
                    phrases.push(p);
                }
            }
        }
        let trie = crate::biasing::BiasTrie::from_phrases(&phrases);
        self.model.set_bias(Some(trie), strength);
    }
}

impl Transcriber for ParakeetTDT {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        // Keep upstream's cached feature extraction (v0.3.6 perf win) AND our
        // confidence return value from forward().
        let features = audio::extract_features_with_cache(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            &self.feature_cache,
        )?;
        let (tokens, frame_indices, durations, confidences) = self.model.forward(features)?;

        let mut result = self.decoder.decode_with_timestamps(
            &tokens,
            &frame_indices,
            &durations,
            &confidences,
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

        Ok(result)
    }
}
