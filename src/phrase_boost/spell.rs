//! Spelling a phrase in the model's vocab pieces.

use super::SkipReason;
use std::collections::HashMap;

pub(crate) const MAX_SPELLINGS_PER_VARIANT: usize = 5;
pub(crate) const MIN_LETTERS: usize = 3;

const WORD_START: char = '\u{2581}'; // ▁

/// Every token sequence the decoder could emit for `phrase` starting a word: for
/// each case variant (as written, all lowercase, first letter uppercased), the
/// segmentations of `▁` + words-joined-by-`▁` into vocab pieces that use the
/// fewest pieces, at most 5 per variant, ordered by token ids, deduplicated
/// across variants. Letters are alphanumeric characters; `TooShort` when there
/// are ≤ 2 of them (so a whitespace-only phrase is always `TooShort`); otherwise
/// `NotSpellable` when no variant segments. Whitespace is trimmed and runs
/// collapse to one word boundary. Punctuation is spelled like any other
/// character, including a bare `▁` piece before a leading `.`.
pub(crate) struct Speller {
    pieces: HashMap<String, u32>,
    max_piece_chars: usize,
}

impl Speller {
    /// Builds the piece map. The blank and every `<…>` piece are never used.
    pub(crate) fn new(id_to_token: &[String], blank_id: usize) -> Speller {
        let mut pieces = HashMap::new();
        let mut max_piece_chars = 0;
        for (id, tok) in id_to_token.iter().enumerate() {
            let special = tok.chars().count() >= 2 && tok.starts_with('<') && tok.ends_with('>');
            if id == blank_id || tok.is_empty() || special {
                continue;
            }
            max_piece_chars = max_piece_chars.max(tok.chars().count());
            pieces.entry(tok.clone()).or_insert(id as u32);
        }
        Speller { pieces, max_piece_chars }
    }

    pub(crate) fn spell(&self, phrase: &str) -> Result<Vec<Vec<u32>>, SkipReason> {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let letters = words
            .iter()
            .flat_map(|w| w.chars())
            .filter(|c| c.is_alphanumeric())
            .count();
        if letters < MIN_LETTERS {
            return Err(SkipReason::TooShort);
        }
        let joined = words.join(&WORD_START.to_string());

        let lower = joined.to_lowercase();
        let mut first_upper = String::new();
        let mut chars = lower.chars();
        if let Some(c) = chars.next() {
            first_upper.extend(c.to_uppercase());
            first_upper.push_str(chars.as_str());
        }

        let mut out: Vec<Vec<u32>> = Vec::new();
        let mut seen_variants: Vec<String> = Vec::new();
        for variant in [joined, lower, first_upper] {
            if seen_variants.contains(&variant) {
                continue;
            }
            let target: Vec<char> = std::iter::once(WORD_START).chain(variant.chars()).collect();
            out.extend(self.fewest_pieces(&target));
            seen_variants.push(variant);
        }
        out.sort();
        out.dedup();
        if out.is_empty() {
            Err(SkipReason::NotSpellable)
        } else {
            Ok(out)
        }
    }

    /// Up to MAX_SPELLINGS_PER_VARIANT minimum-piece segmentations, in id order.
    fn fewest_pieces(&self, target: &[char]) -> Vec<Vec<u32>> {
        let n = target.len();
        // moves[i]: (piece id, next position) for every piece starting at i.
        let mut moves: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
        for (i, m) in moves.iter_mut().enumerate() {
            for len in 1..=self.max_piece_chars.min(n - i) {
                let s: String = target[i..i + len].iter().collect();
                if let Some(&id) = self.pieces.get(&s) {
                    m.push((id, i + len));
                }
            }
            m.sort();
        }
        // best[i]: fewest pieces covering target[i..]; None if impossible.
        let mut best: Vec<Option<usize>> = vec![None; n + 1];
        best[n] = Some(0);
        for i in (0..n).rev() {
            best[i] = moves[i]
                .iter()
                .filter_map(|&(_, j)| best[j].map(|b| b + 1))
                .min();
        }
        let mut found = Vec::new();
        if best[0].is_some() {
            let mut path = Vec::new();
            Self::walk(0, n, &moves, &best, &mut path, &mut found);
        }
        found
    }

    fn walk(
        i: usize,
        n: usize,
        moves: &[Vec<(u32, usize)>],
        best: &[Option<usize>],
        path: &mut Vec<u32>,
        found: &mut Vec<Vec<u32>>,
    ) {
        if found.len() >= MAX_SPELLINGS_PER_VARIANT {
            return;
        }
        if i == n {
            found.push(path.clone());
            return;
        }
        let want = best[i].map(|b| b - 1);
        for &(id, j) in &moves[i] {
            if best[j] == want {
                path.push(id);
                Self::walk(j, n, moves, best, path, found);
                path.pop();
                if found.len() >= MAX_SPELLINGS_PER_VARIANT {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn fixture() -> (Vec<String>, usize) {
        let v = vocab(&[
            "<unk>", "▁C", "▁Ca", "a", "i", "r", "n", "rn", "irn", "▁c", "▁ca", "ir", "▁New",
            "▁York", "▁", ".", "-", "cl", "aude", "env", "<blk>",
        ]);
        let blank = v.len() - 1;
        (v, blank)
    }

    fn id(v: &[String], s: &str) -> u32 {
        v.iter().position(|t| t == s).unwrap() as u32
    }

    #[test]
    fn the_fewest_pieces_win() {
        let (v, b) = fixture();
        let got = Speller::new(&v, b).spell("Cairn").unwrap();
        let want = vec![
            vec![id(&v, "▁Ca"), id(&v, "irn")],
            vec![id(&v, "▁ca"), id(&v, "irn")],
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn every_spelling_starts_at_a_word_start() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        for phrase in ["Cairn", "New York", ".env"] {
            for s in sp.spell(phrase).unwrap() {
                assert!(v[s[0] as usize].starts_with('▁'), "{phrase}: {s:?}");
            }
        }
    }

    #[test]
    fn case_variants_are_spelled_too() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        for p in ["cairn", "CAIRN"] {
            let got = sp.spell(p).unwrap();
            assert!(got.contains(&vec![id(&v, "▁Ca"), id(&v, "irn")]), "{p}");
            assert!(got.contains(&vec![id(&v, "▁ca"), id(&v, "irn")]), "{p}");
        }
    }

    #[test]
    fn a_phrase_of_two_letters_or_fewer_is_skipped() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        for p in ["AI", "Q", ".a", "   ", ""] {
            assert_eq!(sp.spell(p), Err(SkipReason::TooShort), "{p:?}");
        }
    }

    #[test]
    fn an_unspellable_phrase_is_skipped_not_fatal() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        assert_eq!(sp.spell("Zürich"), Err(SkipReason::NotSpellable));
        assert_eq!(sp.spell("abc🙂"), Err(SkipReason::NotSpellable));
    }

    #[test]
    fn a_leading_dot_spells_through_a_bare_word_start() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        assert_eq!(
            sp.spell(".env").unwrap(),
            vec![vec![id(&v, "▁"), id(&v, "."), id(&v, "env")]]
        );
        assert_eq!(
            sp.spell(".claude").unwrap(),
            vec![vec![id(&v, "▁"), id(&v, "."), id(&v, "cl"), id(&v, "aude")]]
        );
    }

    #[test]
    fn special_pieces_are_never_used() {
        // "<abc>" is special; with blank = "▁abc" (id 2) only the 2-piece route is left.
        let v = vocab(&["<unk>", "▁ab", "▁abc", "<abc>", "c"]);
        let got = Speller::new(&v, 2).spell("abc").unwrap();
        assert_eq!(got, vec![vec![1, 4]]);
        assert!(got.iter().flatten().all(|&t| t != 2 && t != 3 && t != 0));
        // And with only specials/blank available the phrase is unspellable.
        let v = vocab(&["<unk>", "▁abc", "<abc>"]);
        assert_eq!(Speller::new(&v, 1).spell("abc"), Err(SkipReason::NotSpellable));
    }

    #[test]
    fn a_literal_special_piece_is_not_matched_by_its_text() {
        // Without the `<...>` filter this would spell as [▁, <unk>].
        let v = vocab(&["▁", "<unk>", "<blk>"]);
        assert_eq!(Speller::new(&v, 2).spell("<unk>"), Err(SkipReason::NotSpellable));
    }

    #[test]
    fn a_multi_word_phrase_spans_the_space() {
        let (v, b) = fixture();
        let sp = Speller::new(&v, b);
        assert_eq!(
            sp.spell("New York").unwrap(),
            vec![vec![id(&v, "▁New"), id(&v, "▁York")]]
        );
        assert_eq!(sp.spell("  New \t York ").unwrap(), sp.spell("New York").unwrap());
    }

    #[test]
    fn spellings_are_capped_and_ordered() {
        // "▁abcdefgh" splits into two pieces 7 ways (prefix + suffix).
        let v = vocab(&[
            "▁a", "▁ab", "▁abc", "▁abcd", "▁abcde", "▁abcdef", "▁abcdefg", "bcdefgh", "cdefgh",
            "defgh", "efgh", "fgh", "gh", "h",
        ]);
        let got = Speller::new(&v, 99).spell("abcdefgh").unwrap();
        assert_eq!(got.len(), MAX_SPELLINGS_PER_VARIANT);
        assert!(got.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(got[0], vec![0, 7]);
        assert_eq!(got[4], vec![4, 11]);
    }

    /// The five punctuation phrases against the real model vocab, when installed.
    #[test]
    fn punctuation_phrases_spell_in_the_real_vocab() {
        let path =
            r"C:\Users\geisl\AppData\Roaming\speakey\app\data\models\parakeet-tdt-v3\vocab.txt";
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("skipped: vocab not found at {path}");
            return;
        };
        let mut v: Vec<String> = Vec::new();
        for line in text.lines() {
            let (tok, idx) = line.rsplit_once(' ').unwrap();
            let idx: usize = idx.parse().unwrap();
            if v.len() <= idx {
                v.resize(idx + 1, String::new());
            }
            v[idx] = tok.to_string();
        }
        let blank = v.len(); // the blank sits after the listed pieces
        let sp = Speller::new(&v, blank);
        for p in [".claude", ".env", "claude.md", "dockbox-prod", "parakeet-rs"] {
            let got = sp.spell(p).unwrap_or_else(|e| panic!("{p}: {e:?}"));
            assert!(!got.is_empty(), "{p}");
        }
    }
}
