use crate::decoder::TranscriptionResult;
use crate::error::Result;
use crate::model_tdt::EncoderStates;
use crate::vocab::Vocabulary;

/// Seconds at which a token emitted on encoder frame `frame` starts: the one clock shared by
/// word timestamps and phrase-boost hit times.
pub(crate) fn token_start_seconds(frame: usize, hop_length: usize, sample_rate: usize) -> f32 {
    (frame * EncoderStates::ENCODER_STRIDE * hop_length) as f32 / sample_rate as f32
}

/// TDT greedy decoder for Parakeet TDT models
#[derive(Debug)]
pub struct ParakeetTDTDecoder {
    vocab: Vocabulary,
}

impl ParakeetTDTDecoder {
    /// Load decoder from vocab file
    pub fn from_vocab(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    /// The vocabulary token ids index into.
    pub(crate) fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Decode tokens with timestamps
    /// For TDT models, greedy decoding is done in the model, here we just convert to text
    pub fn decode_with_timestamps(
        &self,
        tokens: &[usize],
        frame_indices: &[usize],
        _durations: &[usize],
        confidences: &[f32],
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let mut result_tokens = Vec::new();
        let mut full_text = String::new();

        for (i, &token_id) in tokens.iter().enumerate() {
            if let Some(token_text) = self.vocab.id_to_text(token_id) {
                let frame = frame_indices[i];
                let start = token_start_seconds(frame, hop_length, sample_rate);
                let end = if i + 1 < frame_indices.len() {
                    token_start_seconds(frame_indices[i + 1], hop_length, sample_rate)
                } else {
                    start + 0.01
                };

                // Handle SentencePiece format (▁ prefix for word start)
                let mut display_text = token_text.replace('▁', " ");

                // Insert space before pure digit tokens following words.
                // SentencePiece digits lack the ▁ prefix, causing "at60" instead of "at 60".
                // Skip single uppercase letters (A4, B12) but allow lowercase 'a' (article).
                if !full_text.is_empty()
                    && !display_text.starts_with(' ')
                    && display_text.chars().all(|c| c.is_ascii_digit())
                {
                    let trailing_letters: usize = full_text
                        .chars()
                        .rev()
                        .take_while(|c| c.is_alphabetic())
                        .count();
                    let last_char = full_text.chars().last();
                    let is_article_a = trailing_letters == 1 && last_char == Some('a');
                    if trailing_letters > 1 || is_article_a {
                        display_text.insert(0, ' ');
                    }
                }

                // Skip special tokens, `<unk>` included: it is the model having no spelling for a
                // sound, never text (see `unk_is_never_text_or_a_word`).
                if !(token_text.starts_with('<') && token_text.ends_with('>')) {
                    full_text.push_str(&display_text);

                    let conf = if i < confidences.len() {
                        confidences[i]
                    } else {
                        0.0
                    };
                    result_tokens.push(crate::decoder::TimedToken {
                        text: display_text,
                        start,
                        end,
                        confidence: conf,
                    });
                }
            }
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            tokens: result_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(tokens: &[&str]) -> Vocabulary {
        Vocabulary {
            id_to_token: tokens.iter().map(|s| s.to_string()).collect(),
            _blank_id: 0,
        }
    }

    #[test]
    fn test_digit_spacing_after_word() {
        // Simulates "like 100" tokenized as ["▁like", "1", "0", "0"]
        let vocab = make_vocab(&["▁like", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], &[0.9, 0.8, 0.7, 0.6], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "like 100");
    }

    /// `<unk>` is the model saying it has no spelling for a sound: never text, never a word. Kept
    /// as text it was typed into the user's document (Parakeet Ultra emits it around quoted foreign
    /// words: 73 times in 24 FLEURS clips, 2026-11-23).
    #[test]
    fn unk_is_never_text_or_a_word() {
        let vocab = make_vocab(&["▁the", "<unk>", "▁dog", "<unk>"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], &[0.9, 0.8, 0.7, 0.6], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "the dog");
        assert!(result.tokens.iter().all(|t| !t.text.contains("<unk>")), "{:?}", result.tokens);
        assert_eq!(result.tokens.len(), 2);
    }

    #[test]
    fn test_digit_spacing_after_article_a() {
        // Simulates "a 24" tokenized as ["▁a", "2", "4"]
        let vocab = make_vocab(&["▁a", "2", "4"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2], &[0, 1, 2], &[1, 1, 1], &[0.9, 0.8, 0.7], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "a 24");
    }

    #[test]
    fn test_no_spacing_after_single_uppercase() {
        // Simulates "A4" tokenized as ["▁A", "4"] - should stay together
        let vocab = make_vocab(&["▁A", "4"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1], &[0, 1], &[1, 1], &[0.9, 0.8], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "A4");
    }

    #[test]
    fn test_no_spacing_after_symbol() {
        // Simulates "$100" tokenized as ["$", "1", "0", "0"]
        let vocab = make_vocab(&["$", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], &[0.9, 0.8, 0.7, 0.6], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "$100");
    }

    #[test]
    fn test_spacing_in_phrase() {
        // Simulates "In 2021" tokenized as ["▁In", "2", "0", "2", "1"]
        let vocab = make_vocab(&["▁In", "2", "0", "2", "1"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                &[1, 1, 1, 1, 1],
                &[0.9, 0.8, 0.7, 0.6, 0.5],
                160,
                16000,
            )
            .unwrap();
        assert_eq!(result.text, "In 2021");
    }

    #[test]
    fn test_tokens_have_correct_spacing() {
        // Verify individual token texts have spacing applied
        let vocab = make_vocab(&["▁like", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], &[0.9, 0.8, 0.7, 0.6], 160, 16000)
            .unwrap();

        // Check token texts - first digit should have space prepended
        assert_eq!(result.tokens[0].text, " like");
        assert_eq!(result.tokens[1].text, " 1"); // Space added by heuristic
        assert_eq!(result.tokens[2].text, "0");
        assert_eq!(result.tokens[3].text, "0");
    }

    #[test]
    fn test_confidence_passthrough() {
        // Verify confidence values are passed through from model to tokens
        let vocab = make_vocab(&["▁hello", "▁world"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1], &[0, 1], &[1, 1], &[0.95, 0.42], 160, 16000)
            .unwrap();
        assert_eq!(result.tokens.len(), 2);
        assert!((result.tokens[0].confidence - 0.95).abs() < 1e-6);
        assert!((result.tokens[1].confidence - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_full_flow_with_timestamp_processing() {
        use crate::timestamps::{process_timestamps, TimestampMode};

        let vocab = make_vocab(&["▁like", "1", "0", "0", "▁bucks"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                &[1, 1, 1, 1, 1],
                &[0.9, 0.8, 0.7, 0.6, 0.95],
                160,
                16000,
            )
            .unwrap();

        // Test Words mode (what Undertone likely uses)
        let words = process_timestamps(&result.tokens, TimestampMode::Words);
        let text: String = words
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text, "like 100 bucks");

        // Test Tokens mode
        let tokens_text: String = result.tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(tokens_text.trim(), "like 100 bucks");
    }

    #[test]
    fn hit_times_use_the_decoders_clock() {
        for f in [0usize, 1, 7, 123, 4000] {
            assert_eq!(token_start_seconds(f, 160, 16000), (f * 8 * 160) as f32 / 16000.0);
        }
        // decode_with_timestamps reads the same clock: start of each token, and the next
        // token's start as the end.
        let vocab = make_vocab(&["▁a", "▁b", "▁c"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let r = decoder
            .decode_with_timestamps(&[0, 1, 2], &[3, 17, 40], &[1, 1, 1], &[0.9, 0.9, 0.9], 160, 16000)
            .unwrap();
        assert_eq!(r.tokens[0].start, token_start_seconds(3, 160, 16000));
        assert_eq!(r.tokens[0].end, token_start_seconds(17, 160, 16000));
        assert_eq!(r.tokens[2].start, token_start_seconds(40, 160, 16000));
    }
}
