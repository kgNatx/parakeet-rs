use crate::decoder::TimedToken;

/// Timestamp output mode for transcription results
///
/// Determines how token-level timestamps are grouped and presented:
/// - `Tokens`: Raw token-level output from the model (most detailed)
/// - `Words`: Tokens grouped into individual words
/// - `Sentences`: Tokens grouped by sentence boundaries (., ?, !)
///
/// # Model-Specific Recommendations
///
/// - **Parakeet CTC (English)**: Use `Words` mode. The CTC model only outputs lowercase
///   alphabet without punctuation, so sentence segmentation is not possible.
/// - **Parakeet TDT (Multilingual)**: Use `Sentences` mode. The TDT model predicts
///   punctuation, enabling natural sentence boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampMode {
    /// Raw token-level timestamps from the model
    #[default]
    Tokens,
    /// Word-level timestamps (groups subword tokens)
    Words,
    /// Sentence-level timestamps (groups by punctuation)
    ///
    /// Note: Only works with models that predict punctuation (e.g., Parakeet TDT).
    /// CTC models don't predict punctuation, so use `Words` mode instead.
    Sentences,
}

/// Convert token timestamps to the requested output mode
///
/// Takes raw token-level timestamps from the model and optionally groups them
/// into words or sentences while preserving the original timing information.
///
/// # Arguments
///
/// * `tokens` - Raw token-level timestamps from model output
/// * `mode` - Desired grouping level (Tokens, Words, or Sentences)
///
/// # Returns
///
/// Vector of TimedToken with timestamps at the requested granularity
pub fn process_timestamps(tokens: &[TimedToken], mode: TimestampMode) -> Vec<TimedToken> {
    match mode {
        TimestampMode::Tokens => tokens.to_vec(),
        TimestampMode::Words => group_by_words(tokens),
        TimestampMode::Sentences => group_by_sentences(tokens),
    }
}

// Group tokens into words based on word boundary markers
pub(crate) fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current_word_text = String::new();
    let mut current_word_start = 0.0;
    let mut current_word_min_confidence = 1.0f32;
    let mut last_word_lower = String::new();

    for (i, token) in tokens.iter().enumerate() {
        // Space-only tokens (from SentencePiece ▁ word boundaries) act as word separators
        // but don't contribute text. Save current word if we hit one.
        if token.text.trim().is_empty() {
            if !current_word_text.is_empty() {
                let word_lower = current_word_text.to_lowercase();
                if word_lower != last_word_lower {
                    words.push(TimedToken {
                        text: current_word_text.clone(),
                        start: current_word_start,
                        end: if i > 0 { tokens[i - 1].end } else { token.end },
                        confidence: current_word_min_confidence,
                    });
                    last_word_lower = word_lower;
                }
                current_word_text.clear();
                current_word_min_confidence = 1.0;
            }
            continue;
        }

        // Check if this starts a new word (SentencePiece uses ▁ or space prefix)
        // Also treat PURE punctuation marks (like ".", ",") as separate words
        // But NOT contractions like "'re" or "'s" or hyphenations like "-two" (ex. twenty-two) which should attach to previous word
        let is_pure_punctuation =
            !token.text.is_empty() && token.text.chars().all(|c| c.is_ascii_punctuation());

        // Check if this is a contraction or hyphenation suffix
        // These should NOT start a new word - they attach to the previous word
        let token_without_marker = token.text.trim_start_matches('▁').trim_start_matches(' ');
        let is_contraction = token_without_marker.starts_with('\'');
        let is_hyphenation = token_without_marker.starts_with('-');

        let starts_word =
            (token.text.starts_with('▁') || token.text.starts_with(' ') || is_pure_punctuation)
                && !is_contraction
                && !is_hyphenation
                || i == 0;

        if starts_word && !current_word_text.is_empty() {
            // Save previous word (with deduplication)
            let word_lower = current_word_text.to_lowercase();
            if word_lower != last_word_lower {
                words.push(TimedToken {
                    text: current_word_text.clone(),
                    start: current_word_start,
                    end: tokens[i - 1].end,
                    confidence: current_word_min_confidence,
                });
                last_word_lower = word_lower;
            }
            current_word_text.clear();
            current_word_min_confidence = 1.0;
        }

        // Start new word or append to current
        if current_word_text.is_empty() {
            current_word_start = token.start;
        }

        // Track minimum confidence across tokens in this word
        current_word_min_confidence = current_word_min_confidence.min(token.confidence);

        // Add token text, removing word boundary markers
        let token_text = token.text.trim_start_matches('▁').trim_start_matches(' ');
        current_word_text.push_str(token_text);
    }

    // Add final word
    if !current_word_text.is_empty() {
        let word_lower = current_word_text.to_lowercase();
        if word_lower != last_word_lower {
            words.push(TimedToken {
                text: current_word_text,
                start: current_word_start,
                end: tokens.last().unwrap().end,
                confidence: current_word_min_confidence,
            });
        }
    }

    words
}

// Group words into sentences based on punctuation
fn group_by_sentences(tokens: &[TimedToken]) -> Vec<TimedToken> {
    // First get word-level grouping
    let words = group_by_words(tokens);
    if words.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current_sentence = Vec::new();

    for word in words {
        current_sentence.push(word.clone());

        // Check if word ends with sentence terminator
        let ends_sentence =
            word.text.contains('.') || word.text.contains('?') || word.text.contains('!');

        if ends_sentence {
            let sentence_text = format_sentence(&current_sentence);
            let start = current_sentence.first().unwrap().start;
            let end = current_sentence.last().unwrap().end;
            // Sentence confidence = min across all word confidences
            let min_conf = current_sentence
                .iter()
                .map(|w| w.confidence)
                .fold(1.0f32, f32::min);

            if !sentence_text.is_empty() {
                sentences.push(TimedToken {
                    text: sentence_text,
                    start,
                    end,
                    confidence: min_conf,
                });
            }
            current_sentence.clear();
        }
    }

    // Add final sentence if exists
    if !current_sentence.is_empty() {
        let sentence_text = format_sentence(&current_sentence);
        let start = current_sentence.first().unwrap().start;
        let end = current_sentence.last().unwrap().end;
        let min_conf = current_sentence
            .iter()
            .map(|w| w.confidence)
            .fold(1.0f32, f32::min);

        if !sentence_text.is_empty() {
            sentences.push(TimedToken {
                text: sentence_text,
                start,
                end,
                confidence: min_conf,
            });
        }
    }

    sentences
}

// Join words with punctuation spacing
fn format_sentence(words: &[TimedToken]) -> String {
    let result: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();

    // Join words, but don't add space before certain punctuation
    let mut output = String::new();
    for (i, word) in result.iter().enumerate() {
        // Check if this word is standalone punctuation that shouldn't have space before it
        // Contractions like "'re" or "'s" should have spaces before them
        let is_standalone_punct = word.len() == 1
            && word
                .chars()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));

        if i > 0 && !is_standalone_punct {
            output.push(' ');
        }
        output.push_str(word);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_grouping() {
        let tokens = vec![
            TimedToken {
                text: "▁Hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.9,
            },
            TimedToken {
                text: "▁world".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 0.8,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn test_word_grouping_with_hyphenated_word() {
        let tokens = vec![
            TimedToken {
                text: "▁twenty".to_string(),
                start: 0.0,
                end: 0.3,
                confidence: 0.9,
            },
            TimedToken {
                text: "-two".to_string(),
                start: 0.3,
                end: 0.6,
                confidence: 0.7,
            },
            TimedToken {
                text: "▁apples".to_string(),
                start: 0.6,
                end: 1.0,
                confidence: 0.95,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "twenty-two");
        assert_eq!(words[1].text, "apples");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.6);
        assert_eq!(words[1].start, 0.6);
        assert_eq!(words[1].end, 1.0);
        // Hyphenated word confidence = min(0.9, 0.7) = 0.7
        assert!((words[0].confidence - 0.7).abs() < 1e-6);
        assert!((words[1].confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_sentence_grouping() {
        let tokens = vec![
            TimedToken {
                text: "▁Hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.9,
            },
            TimedToken {
                text: "▁world".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 0.8,
            },
            TimedToken {
                text: ".".to_string(),
                start: 1.0,
                end: 1.1,
                confidence: 0.99,
            },
        ];

        let sentences = group_by_sentences(&tokens);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "Hello world.");
        assert_eq!(sentences[0].start, 0.0);
        assert_eq!(sentences[0].end, 1.1);
        // Sentence confidence = min across words = min(0.9, 0.8, 0.99) = 0.8
        assert!((sentences[0].confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_preservation() {
        let words = vec![
            TimedToken {
                text: "uh".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: "uh".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 1.0,
            },
            TimedToken {
                text: "hello".to_string(),
                start: 1.0,
                end: 1.5,
                confidence: 1.0,
            },
        ];

        let result = format_sentence(&words);
        assert_eq!(result, "uh uh hello");
    }

    #[test]
    fn test_space_token_separates_words_from_digits() {
        // Simulates "like 100" tokenized as [" like", " ", "1", "0", "0"]
        // The space-only token should act as word boundary
        let tokens = vec![
            TimedToken {
                text: " like".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.9,
            },
            TimedToken {
                text: " ".to_string(), // Space-only token from ▁
                start: 0.5,
                end: 0.5,
                confidence: 0.0,
            },
            TimedToken {
                text: "1".to_string(),
                start: 0.5,
                end: 0.6,
                confidence: 0.8,
            },
            TimedToken {
                text: "0".to_string(),
                start: 0.6,
                end: 0.7,
                confidence: 0.7,
            },
            TimedToken {
                text: "0".to_string(),
                start: 0.7,
                end: 0.8,
                confidence: 0.6,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "like");
        assert_eq!(words[1].text, "100");

        // Also test sentence formatting
        let sentence = format_sentence(&words);
        assert_eq!(sentence, "like 100");
    }

    #[test]
    fn test_word_confidence_is_min_of_tokens() {
        // "hello" composed of 3 sub-tokens: confidence should be min
        let tokens = vec![
            TimedToken {
                text: "▁hel".to_string(),
                start: 0.0,
                end: 0.2,
                confidence: 0.95,
            },
            TimedToken {
                text: "lo".to_string(),
                start: 0.2,
                end: 0.4,
                confidence: 0.42,
            },
            TimedToken {
                text: "▁world".to_string(),
                start: 0.4,
                end: 0.8,
                confidence: 0.88,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        // min(0.95, 0.42) = 0.42
        assert!((words[0].confidence - 0.42).abs() < 1e-6);
        assert!((words[1].confidence - 0.88).abs() < 1e-6);
    }
}
