use std::collections::HashMap;

/// Greedy longest-match tokenization over a SentencePiece piece list.
/// Prepends the `▁` word-start marker, then repeatedly consumes the longest
/// vocab piece matching the remaining prefix. Chars with no matching piece are
/// skipped (best-effort — approximate encoder, validated in the harness).
pub fn greedy_tokenize(word: &str, piece_to_id: &HashMap<String, u32>) -> Vec<u32> {
    let s: Vec<char> = format!("\u{2581}{}", word).chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut matched: Option<(u32, usize)> = None;
        // Longest match first.
        for len in (1..=(s.len() - i)).rev() {
            let cand: String = s[i..i + len].iter().collect();
            if let Some(&id) = piece_to_id.get(&cand) {
                matched = Some((id, len));
                break;
            }
        }
        match matched {
            Some((id, len)) => {
                out.push(id);
                i += len;
            }
            None => i += 1,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_vocab() -> HashMap<String, u32> {
        // Pieces resembling SentencePiece output.
        [("\u{2581}en", 10u32), ("tan", 11), ("gram", 12), ("\u{2581}the", 13), ("s", 14)]
            .into_iter().map(|(s, i)| (s.to_string(), i)).collect()
    }

    #[test]
    fn tokenizes_multi_piece_word() {
        let v = toy_vocab();
        // "entangram" -> ▁en | tan | gram
        assert_eq!(greedy_tokenize("entangram", &v), vec![10, 11, 12]);
    }

    #[test]
    fn tokenizes_single_piece_word() {
        let v = toy_vocab();
        assert_eq!(greedy_tokenize("the", &v), vec![13]);
    }

    #[test]
    fn skips_unmatched_chars_without_panic() {
        let v = toy_vocab();
        // 'z' has no piece; matcher skips it, still returns the matchable pieces.
        assert_eq!(greedy_tokenize("enz", &v), vec![10]);
    }
}
