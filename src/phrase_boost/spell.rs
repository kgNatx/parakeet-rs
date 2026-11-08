//! Spelling a phrase in the model's vocab pieces.

use super::SkipReason;
use crate::error::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) const MIN_LETTERS: usize = 3;

/// Loads a `tokenizer.json`.
pub(crate) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| {
        Error::Tokenizer(format!("failed to load tokenizer {}: {e}", path.display()))
    })
}

/// Spells a phrase as the token ids the decoder would emit for it starting a
/// word: the model's canonical tokenizer encoding (as NeMo spells boosted
/// phrases), once per case variant (as written, all lowercase, first letter
/// uppercased), sorted by token ids and deduplicated, so at most three.
/// Letters are alphanumeric characters; `TooShort` when there are ≤ 2 of them
/// (so a whitespace-only phrase is always `TooShort`). A variant whose encoding
/// is empty or holds the blank or any `<…>` piece (`<unk>`, byte fallback, a
/// control token) is no spelling; `NotSpellable` when no variant spells.
/// Whitespace is trimmed and runs collapse to one space before encoding.
pub(crate) struct Speller {
    tokenizer: Tokenizer,
    /// Per token id: may this piece appear in a spelling?
    usable: Vec<bool>,
}

impl Speller {
    /// `Err(Error::Config)` naming the first mismatching id unless `tokenizer`
    /// holds exactly `id_to_token`'s pieces at the same ids. The blank's id is
    /// exempt: NVIDIA's tokenizer names it `<blank>` where `vocab.txt` says
    /// `<blk>`, and a spelling never uses it.
    pub(crate) fn new(
        tokenizer: Tokenizer,
        id_to_token: &[String],
        blank_id: usize,
    ) -> Result<Speller> {
        let end = tokenizer
            .get_vocab(true)
            .into_values()
            .map(|id| id as usize + 1)
            .chain([id_to_token.len()])
            .max()
            .unwrap_or(0);
        let mismatch = (0..end).filter(|&id| id != blank_id).find_map(|id| {
            let in_tokenizer = tokenizer.id_to_token(id as u32);
            let in_model = id_to_token.get(id).cloned();
            (in_tokenizer != in_model).then_some((id, in_tokenizer, in_model))
        });
        if let Some((id, in_tokenizer, in_model)) = mismatch {
            let show = |p: Option<String>| p.map_or("absent".to_string(), |p| format!("{p:?}"));
            return Err(Error::Config(format!(
                "tokenizer does not match the model vocabulary: id {id} is {} in the \
                 tokenizer but {} in the model",
                show(in_tokenizer),
                show(in_model),
            )));
        }
        let usable = id_to_token
            .iter()
            .enumerate()
            .map(|(id, tok)| {
                let special =
                    tok.chars().count() >= 2 && tok.starts_with('<') && tok.ends_with('>');
                id != blank_id && !tok.is_empty() && !special
            })
            .collect();
        Ok(Speller { tokenizer, usable })
    }

    pub(crate) fn spell(&self, phrase: &str) -> std::result::Result<Vec<Vec<u32>>, SkipReason> {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let letters = words
            .iter()
            .flat_map(|w| w.chars())
            .filter(|c| c.is_alphanumeric())
            .count();
        if letters < MIN_LETTERS {
            return Err(SkipReason::TooShort);
        }
        let joined = words.join(" ");

        let lower = joined.to_lowercase();
        let mut first_upper = String::new();
        let mut chars = lower.chars();
        if let Some(c) = chars.next() {
            first_upper.extend(c.to_uppercase());
            first_upper.push_str(chars.as_str());
        }

        let mut out: Vec<Vec<u32>> = [joined, lower, first_upper]
            .iter()
            .filter_map(|variant| self.encode(variant))
            .collect();
        out.sort();
        out.dedup();
        if out.is_empty() {
            Err(SkipReason::NotSpellable)
        } else {
            Ok(out)
        }
    }

    /// The canonical encoding of `text`, if every piece of it is usable.
    fn encode(&self, text: &str) -> Option<Vec<u32>> {
        let ids = self.tokenizer.encode(text, false).ok()?.get_ids().to_vec();
        let usable = |&id: &u32| self.usable.get(id as usize).copied().unwrap_or(false);
        (!ids.is_empty() && ids.iter().all(usable)).then_some(ids)
    }
}

/// Test fixtures: a tokenizer for a tiny vocabulary.
#[cfg(test)]
pub(crate) mod fixture {
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    /// A tokenizer whose pieces are exactly `id_to_token` without the blank, at
    /// the same ids: Metaspace (`▁`, prepended always, split on spaces) like the
    /// real one, but a greedy longest-prefix model (WordPiece with no
    /// continuation prefix), so a test vocabulary needs no merge pieces. A word
    /// with a character no piece covers encodes as `<unk>`.
    pub(crate) fn tokenizer(id_to_token: &[String], blank_id: usize) -> Tokenizer {
        let vocab: serde_json::Map<String, serde_json::Value> = id_to_token
            .iter()
            .enumerate()
            .filter(|&(id, _)| id != blank_id)
            .map(|(id, piece)| (piece.clone(), serde_json::Value::from(id)))
            .collect();
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Metaspace", "replacement": "\u{2581}",
                "prepend_scheme": "always", "split": true
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordPiece", "unk_token": "<unk>",
                "continuing_subword_prefix": "", "max_input_chars_per_word": 100,
                "vocab": vocab
            }
        });
        Tokenizer::from_str(&json.to_string()).expect("fixture tokenizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_VOCAB: &str =
        r"C:\Users\geisl\AppData\Roaming\speakey\app\data\models\parakeet-tdt-v3\vocab.txt";
    const REAL_TOKENIZER: &str = r"E:\speech-eval\tokenizer-nvidia-v3\tokenizer.json";

    fn vocab(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn speller(v: &[String], blank: usize) -> Speller {
        Speller::new(fixture::tokenizer(v, blank), v, blank).unwrap()
    }

    fn small() -> (Vec<String>, usize) {
        let v = vocab(&[
            "<unk>", "▁C", "▁Ca", "a", "i", "r", "n", "rn", "irn", "▁c", "▁ca", "ir", "▁New",
            "▁York", "<blk>",
        ]);
        let blank = v.len() - 1;
        (v, blank)
    }

    fn id(v: &[String], s: &str) -> u32 {
        v.iter().position(|t| t == s).unwrap() as u32
    }

    /// The real model vocabulary and NVIDIA's tokenizer for it, when both are present.
    fn real() -> Option<(Vec<String>, usize, Speller)> {
        let (vp, tp) = (Path::new(REAL_VOCAB), Path::new(REAL_TOKENIZER));
        if !vp.exists() || !tp.exists() {
            eprintln!("skipped: needs {REAL_VOCAB} and {REAL_TOKENIZER}");
            return None;
        }
        let v = crate::vocab::Vocabulary::from_file(vp).unwrap().id_to_token;
        let blank = v.len() - 1;
        let sp = Speller::new(load_tokenizer(tp).unwrap(), &v, blank).unwrap();
        Some((v, blank, sp))
    }

    fn as_pieces(v: &[String], spellings: &[Vec<u32>]) -> Vec<String> {
        spellings
            .iter()
            .map(|s| s.iter().map(|&t| v[t as usize].as_str()).collect::<Vec<_>>().join("|"))
            .collect()
    }

    /// Values from the 2026-11-07 Python check against the real tokenizer.json:
    /// the canonical BPE encoding, not the fewest pieces (`▁bel|ly`).
    #[test]
    fn phrases_spell_as_the_canonical_tokenizer_encodes_them() {
        let Some((v, _, sp)) = real() else { return };
        for (phrase, want) in [
            ("belly", "▁b|ell|y"),
            ("chapel", "▁ch|ap|el"),
            ("kneeling", "▁k|ne|el|ing"),
            ("Covalex", "▁C|ov|ale|x"),
            ("dockbox-prod", "▁do|ck|bo|x|-|pro|d"),
            (".claude", "▁|.|c|la|ude"),
        ] {
            let got = as_pieces(&v, &sp.spell(phrase).unwrap());
            assert!(got.iter().any(|s| s == want), "{phrase}: {got:?}");
        }
    }

    #[test]
    fn one_spelling_per_case_variant_deduplicated() {
        let Some((v, _, sp)) = real() else { return };
        // As written, lowercase and first-upper: `Covalex` is its own first-upper form.
        let got = as_pieces(&v, &sp.spell("Covalex").unwrap());
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got.contains(&"▁C|ov|ale|x".to_string()), "{got:?}");
        // All three variants coincide for a lowercase word whose first-upper form
        // encodes differently; never more than three spellings.
        for p in ["belly", "BELLY", "Belly", "dockbox-prod", "New York"] {
            let got = sp.spell(p).unwrap();
            assert!((1..=3).contains(&got.len()), "{p}: {got:?}");
            let mut d = got.clone();
            d.dedup();
            assert_eq!(d, got, "{p}");
        }
        // Casings of one phrase spell to the same set.
        assert_eq!(sp.spell("belly").unwrap(), sp.spell("Belly").unwrap());
    }

    #[test]
    fn a_phrase_needing_byte_fallback_or_unk_is_not_spellable() {
        let Some((_, _, sp)) = real() else { return };
        for p in ["abc🙂", "東京都", "<unk>", "<|endoftext|>"] {
            assert_eq!(sp.spell(p), Err(SkipReason::NotSpellable), "{p}");
        }
    }

    #[test]
    fn the_real_tokenizer_matches_the_real_vocab() {
        let Some((v, blank, _)) = real() else { return };
        assert_eq!(v[blank], "<blk>");
    }

    #[test]
    fn a_tokenizer_for_another_vocabulary_is_refused() {
        let (v, blank) = small();
        let t = fixture::tokenizer(&v, blank);
        // One piece renamed, same size: the error names it.
        let mut renamed = v.clone();
        renamed[4] = "o".to_string();
        let err = Speller::new(t.clone(), &renamed, blank).err().expect("refused");
        let msg = err.to_string();
        assert!(msg.contains("id 4") && msg.contains("\"i\"") && msg.contains("\"o\""), "{msg}");
        // A model vocabulary with a piece the tokenizer lacks.
        let mut longer = v.clone();
        longer.insert(blank, "z".to_string());
        assert!(Speller::new(t.clone(), &longer, blank + 1).is_err());
        // A tokenizer with a piece past the model's end.
        let with_extra = fixture::tokenizer(&v, usize::MAX); // `<blk>` as an ordinary piece
        let err = Speller::new(with_extra, &v[..blank], usize::MAX).err().expect("refused");
        assert!(err.to_string().contains(&format!("id {blank} is \"<blk>\"")), "{err}");
        assert!(err.to_string().contains("absent in the model"), "{err}");
        // The blank's id is exempt: any piece, or none, may sit there.
        let mut renamed_blank = v.clone();
        renamed_blank[blank] = "<blank>".to_string();
        assert!(Speller::new(t.clone(), &renamed_blank, blank).is_ok());
        assert!(Speller::new(fixture::tokenizer(&v, usize::MAX), &v, blank).is_ok());
        // And the real tokenizer against a tiny vocabulary.
        if Path::new(REAL_TOKENIZER).exists() {
            let real = load_tokenizer(Path::new(REAL_TOKENIZER)).unwrap();
            assert!(Speller::new(real, &v, blank).is_err());
        }
    }

    #[test]
    fn a_missing_tokenizer_file_is_an_error() {
        assert!(load_tokenizer(Path::new("no/such/tokenizer.json")).is_err());
    }

    #[test]
    fn every_spelling_starts_at_a_word_start() {
        let (v, b) = small();
        let sp = speller(&v, b);
        for phrase in ["Cairn", "New York"] {
            for s in sp.spell(phrase).unwrap() {
                assert!(v[s[0] as usize].starts_with('▁'), "{phrase}: {s:?}");
            }
        }
    }

    #[test]
    fn case_variants_are_spelled_too() {
        let (v, b) = small();
        let sp = speller(&v, b);
        for p in ["cairn", "CAIRN"] {
            let got = sp.spell(p).unwrap();
            assert!(got.contains(&vec![id(&v, "▁Ca"), id(&v, "irn")]), "{p}");
            assert!(got.contains(&vec![id(&v, "▁ca"), id(&v, "irn")]), "{p}");
        }
    }

    #[test]
    fn a_phrase_of_two_letters_or_fewer_is_skipped() {
        let (v, b) = small();
        let sp = speller(&v, b);
        for p in ["AI", "Q", ".a", "   ", ""] {
            assert_eq!(sp.spell(p), Err(SkipReason::TooShort), "{p:?}");
        }
    }

    #[test]
    fn an_unspellable_phrase_is_skipped_not_fatal() {
        let (v, b) = small();
        let sp = speller(&v, b);
        assert_eq!(sp.spell("Zürich"), Err(SkipReason::NotSpellable));
        assert_eq!(sp.spell("abc🙂"), Err(SkipReason::NotSpellable));
    }

    #[test]
    fn a_literal_special_piece_is_not_matched_by_its_text() {
        // An encoding that yields a `<…>` piece is never a spelling.
        let v = vocab(&["▁", "<unk>", "<blk>"]);
        assert_eq!(speller(&v, 2).spell("<unk>"), Err(SkipReason::NotSpellable));
    }

    #[test]
    fn a_multi_word_phrase_spans_the_space() {
        let (v, b) = small();
        let sp = speller(&v, b);
        assert_eq!(
            sp.spell("New York").unwrap(),
            vec![vec![id(&v, "▁New"), id(&v, "▁York")]]
        );
        assert_eq!(sp.spell("  New \t York ").unwrap(), sp.spell("New York").unwrap());
    }
}
