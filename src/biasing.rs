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

#[derive(Debug, Default, Clone)]
pub struct BiasNode {
    /// token_id -> child node index
    pub children: HashMap<u32, usize>,
    pub is_end: bool,
}

/// Prefix trie of tokenized dictionary phrases for contextual biasing.
#[derive(Debug, Clone)]
pub struct BiasTrie {
    nodes: Vec<BiasNode>,
}

impl BiasTrie {
    pub const ROOT: usize = 0;

    pub fn from_phrases(phrases: &[Vec<u32>]) -> Self {
        let mut nodes = vec![BiasNode::default()]; // ROOT
        for phrase in phrases {
            if phrase.is_empty() {
                continue;
            }
            let mut node = Self::ROOT;
            for &tok in phrase {
                node = match nodes[node].children.get(&tok) {
                    Some(&next) => next,
                    None => {
                        let idx = nodes.len();
                        nodes.push(BiasNode::default());
                        nodes[node].children.insert(tok, idx);
                        idx
                    }
                };
            }
            nodes[node].is_end = true;
        }
        Self { nodes }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1 && self.nodes[Self::ROOT].children.is_empty()
    }

    pub fn children_of(&self, node: usize) -> &HashMap<u32, usize> {
        &self.nodes[node].children
    }

    /// Next active set: always ROOT (new word-starts) plus any child reached by
    /// `token` from a currently-active node.
    pub fn advance(&self, active: &[usize], token: u32) -> Vec<usize> {
        let mut next = vec![Self::ROOT];
        for &node in active {
            if let Some(&child) = self.nodes[node].children.get(&token) {
                if child != Self::ROOT {
                    next.push(child);
                }
            }
        }
        next
    }
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

    #[test]
    fn trie_biases_only_on_path() {
        // Two phrases sharing a prefix: [10,11,12] and [10,11,99].
        let trie = BiasTrie::from_phrases(&[vec![10, 11, 12], vec![10, 11, 99]]);
        // From root, only word-start piece 10 is a child.
        let root_children: Vec<u32> = trie.children_of(BiasTrie::ROOT).keys().copied().collect();
        assert_eq!(root_children, vec![10]);

        // Advance root by 10 -> active includes the node after 10 (plus root).
        let a1 = trie.advance(&[BiasTrie::ROOT], 10);
        // After 10, its child is 11.
        let a1_children: Vec<u32> = a1.iter()
            .flat_map(|&n| trie.children_of(n).keys().copied())
            .collect();
        assert!(a1_children.contains(&11)); // on-path continuation offered
        assert!(a1_children.contains(&10)); // root still offers word-starts
    }

    #[test]
    fn trie_advance_off_path_resets_to_root_only() {
        let trie = BiasTrie::from_phrases(&[vec![10, 11, 12]]);
        // Emit a token (42) that is not a valid continuation -> back to [ROOT].
        let next = trie.advance(&[BiasTrie::ROOT], 42);
        assert_eq!(next, vec![BiasTrie::ROOT]);
    }
}
