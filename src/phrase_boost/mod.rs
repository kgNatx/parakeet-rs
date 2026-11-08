//! Phrase boosting for the greedy TDT decoder.
//!
//! Knows how a list of phrases biases a decode: how a phrase is spelled in the
//! model's vocab pieces, the boosting tree, and the per-step token pick.

mod spell;
mod tree;

#[cfg(test)]
pub(crate) use spell::fixture::tokenizer as fixture_tokenizer;

use crate::error::{Error, Result};
use crate::vocab::Vocabulary;
use crate::model_tdt::argmax_last;
use spell::Speller;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;
use tree::{NodeId, Tree, ROOT};

/// Why a phrase was left out of the boost list. Never fatal: the caller logs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Two letters or fewer: too ambiguous to boost.
    TooShort,
    /// The tokenizer cannot spell it in the model's pieces: its encoding needs
    /// `<unk>`, a byte-fallback piece or another `<…>` piece.
    NotSpellable,
    /// Its strength is 0 (or negative, or NaN, which clamp to 0): nothing to boost.
    NoStrength,
}

/// One phrase to boost and how strongly, in nats per unit of NeMo tree score
/// (research notes 1.6). Strengths are clamped to `[0, PhraseBooster::MAX_STRENGTH]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostPhrase {
    pub text: String,
    pub strength: f32,
}

impl BoostPhrase {
    pub fn new(text: impl Into<String>, strength: f32) -> BoostPhrase {
        BoostPhrase { text: text.into(), strength }
    }
}

/// A boosted phrase the decoder emitted.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostHit {
    pub phrase: String,
    /// Seconds.
    pub start: f32,
    pub end: f32,
    /// Whether boosting changed any of its tokens.
    pub changed: bool,
}

/// What boosting did to one decode.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoostOutcome {
    pub hits: Vec<BoostHit>,
    /// Emitted tokens that differ from the unbiased argmax.
    pub changed_tokens: usize,
    /// Every emitted token id, for the bench probe.
    pub token_ids: Vec<usize>,
}

/// NeMo's boosting-tree defaults (research notes 3.1); not swept.
const C0: f32 = 1.0;
const BETA: f32 = 2.0;

/// A list of phrases, each with its own strength, compiled for one model vocabulary.
///
/// Built once (about a millisecond), immutable, and shared by every decode
/// that should be boosted; each decode walks it with its own `cursor()`.
/// Phrases are deduplicated case-insensitively: one entry, named by the
/// first casing, carrying the spellings of every casing given and the largest
/// of their strengths. Where two phrases' spellings share their first pieces,
/// those pieces are boosted at the larger strength until the spellings part. A
/// phrase that cannot be boosted is kept in `skipped()` with its reason, never
/// silently lost.
pub struct PhraseBooster {
    /// Boosted phrases, strengths clamped.
    phrases: Vec<BoostPhrase>,
    /// Per entry of `phrases`, its spellings as piece strings.
    spellings: Vec<Vec<Vec<String>>>,
    skipped: Vec<(String, SkipReason)>,
    blank_id: usize,
    tree: Tree,
    vocab_len: usize,
    vocab_fingerprint: u64,
    /// The vocabulary's pieces, id -> piece, for reading emitted token ids back.
    pieces: Vec<String>,
}

impl PhraseBooster {
    /// NeMo-Speech.cpp's greedy-safe ceiling for the strength.
    pub const MAX_STRENGTH: f32 = 5.0;

    /// Each strength is clamped to `[0, MAX_STRENGTH]`; NaN counts as 0, and a
    /// phrase at 0 is skipped (`SkipReason::NoStrength`). `tokenizer` spells the phrases and must hold exactly `id_to_token`'s
    /// pieces, blank excepted, at the same ids (`Error::Config` otherwise).
    pub(crate) fn new(
        id_to_token: &[String],
        blank_id: usize,
        tokenizer: Tokenizer,
        phrases: &[BoostPhrase],
    ) -> Result<PhraseBooster> {
        let speller = Speller::new(tokenizer, id_to_token, blank_id)?;
        Ok(Self::compile(id_to_token, blank_id, Some(&speller), phrases))
    }

    /// `new` once the speller is built; `speller` is `None` only for an empty
    /// `phrases`, which never needs spelling.
    fn compile(
        id_to_token: &[String],
        blank_id: usize,
        speller: Option<&Speller>,
        phrases: &[BoostPhrase],
    ) -> PhraseBooster {
        debug_assert!(speller.is_some() || phrases.is_empty());
        // One entry per case-insensitive key, named by its first casing, at the
        // largest strength given; every casing given contributes its spellings
        // (an as-written variant can reach pieces the others cannot).
        let mut groups: Vec<(&String, f32, Vec<&String>)> = Vec::new();
        let mut group_of: HashMap<String, usize> = HashMap::new();
        for phrase in phrases {
            let strength = clamp_strength(phrase.strength);
            match group_of.entry(dedup_key(&phrase.text)) {
                Entry::Occupied(e) => {
                    let g = &mut groups[*e.get()];
                    g.1 = g.1.max(strength);
                    g.2.push(&phrase.text);
                }
                Entry::Vacant(e) => {
                    e.insert(groups.len());
                    groups.push((&phrase.text, strength, vec![&phrase.text]));
                }
            }
        }
        let mut kept = Vec::new();
        let mut spellings = Vec::new();
        let mut skipped = Vec::new();
        let mut seqs: Vec<(Vec<u32>, u32, f32)> = Vec::new();
        for (phrase, strength, casings) in groups {
            let mut ss: Vec<Vec<u32>> = Vec::new();
            let mut first_err = None;
            for casing in casings {
                match speller.map_or(Err(SkipReason::NotSpellable), |s| s.spell(casing)) {
                    Ok(found) => ss.extend(found.into_iter().filter(|s| !s.is_empty())),
                    Err(reason) => {
                        first_err.get_or_insert(reason);
                    }
                }
            }
            ss.sort();
            ss.dedup();
            // A spelling problem is reported before a zero strength: it is the
            // thing that would still stop the phrase at any strength.
            if !ss.is_empty() && strength == 0.0 {
                skipped.push((phrase.clone(), SkipReason::NoStrength));
                continue;
            }
            if ss.is_empty() {
                let reason = first_err.unwrap_or(SkipReason::NotSpellable);
                skipped.push((phrase.clone(), reason));
                continue;
            }
            let index = kept.len() as u32;
            spellings.push(
                ss.iter()
                    .map(|s| {
                        s.iter()
                            .map(|&id| id_to_token[id as usize].clone())
                            .collect()
                    })
                    .collect(),
            );
            seqs.extend(ss.into_iter().map(|s| (s, index, strength)));
            kept.push(BoostPhrase::new(phrase.clone(), strength));
        }
        PhraseBooster {
            phrases: kept,
            spellings,
            skipped,
            blank_id,
            tree: Tree::build(&seqs, C0, BETA),
            vocab_len: id_to_token.len(),
            vocab_fingerprint: fingerprint(id_to_token),
            pieces: id_to_token.to_vec(),
        }
    }

    /// Compile `phrases` for `vocab`, spelled by the `tokenizer.json` at `tokenizer`. The blank id
    /// is the decode loop's own ([`crate::model_tdt::tdt_blank_id`]), of the model's `vocab_size`,
    /// which is `vocab.size()`. An empty `phrases` never reads `tokenizer`, so
    /// "no custom words" needs no tokenizer file; its booster still runs the real cursor.
    pub(crate) fn from_vocabulary(
        vocab: &Vocabulary,
        tokenizer: &Path,
        phrases: &[BoostPhrase],
    ) -> Result<PhraseBooster> {
        let blank_id = crate::model_tdt::tdt_blank_id(vocab.size());
        if phrases.is_empty() {
            return Ok(Self::compile(&vocab.id_to_token, blank_id, None, phrases));
        }
        let tokenizer = spell::load_tokenizer(tokenizer)?;
        PhraseBooster::new(&vocab.id_to_token, blank_id, tokenizer, phrases)
    }

    /// Compile `phrases` for the vocabulary in a model dir's `vocab.txt`, spelled by the
    /// model's canonical `tokenizer.json`, without loading any model. The result equals
    /// `ParakeetTDT::phrase_booster` for the same files. A tokenizer whose pieces are not the
    /// vocabulary's (blank excepted) is an `Error::Config` naming the first mismatch.
    pub fn from_files(
        vocab: &Path,
        tokenizer: &Path,
        phrases: &[BoostPhrase],
    ) -> Result<PhraseBooster> {
        Self::from_vocabulary(&Vocabulary::from_file(vocab)?, tokenizer, phrases)
    }

    /// The vocabulary this booster was built for, `id -> piece`.
    pub fn pieces(&self) -> &[String] {
        &self.pieces
    }

    /// The boosted phrases with their clamped strengths, in first-seen order.
    /// `RawHit::phrase` indexes this.
    pub fn phrases(&self) -> &[BoostPhrase] {
        &self.phrases
    }

    /// Phrases left out, in first-seen order, each with why.
    pub fn skipped(&self) -> &[(String, SkipReason)] {
        &self.skipped
    }

    /// True when no phrase is boosted. An empty booster picks as plain argmax.
    pub fn is_empty(&self) -> bool {
        self.phrases.is_empty()
    }

    /// Number of distinct phrase-first tokens: the tokens that carry a bonus at every step.
    pub fn root_fan_out(&self) -> usize {
        self.tree.root_fan_out()
    }

    /// The spellings of a boosted phrase as piece strings, in token-id order;
    /// empty when the phrase is not boosted. Matched as deduplication matches:
    /// case and whitespace runs are ignored.
    pub fn spellings(&self, phrase: &str) -> Vec<Vec<String>> {
        let key = dedup_key(phrase);
        self.phrases
            .iter()
            .position(|p| dedup_key(&p.text) == key)
            .map(|i| self.spellings[i].clone())
            .unwrap_or_default()
    }

    /// `Err(Error::Config)` unless `id_to_token` is the vocabulary this booster
    /// was built from: the same length and the same piece strings. Token ids
    /// from another vocabulary would boost unrelated pieces, even at the same size.
    pub(crate) fn check_vocab(&self, id_to_token: &[String]) -> crate::error::Result<()> {
        let fp = fingerprint(id_to_token);
        if id_to_token.len() == self.vocab_len && fp == self.vocab_fingerprint {
            return Ok(());
        }
        Err(Error::Config(format!(
            "phrase booster was built for another vocabulary ({} pieces, fingerprint {:016x}); \
             this model has {} pieces, fingerprint {:016x}",
            self.vocab_len,
            self.vocab_fingerprint,
            id_to_token.len(),
            fp
        )))
    }

    /// A fresh per-decode cursor at the tree's root. An empty booster still
    /// gives a real cursor, whose picks are plain argmax.
    pub(crate) fn cursor(&self) -> BoostCursor<'_> {
        BoostCursor {
            booster: self,
            state: ROOT,
            last_changed: false,
            changed: Vec::new(),
            hits: Vec::new(),
        }
    }
}

/// `[0, MAX_STRENGTH]`; NaN is 0.
fn clamp_strength(strength: f32) -> f32 {
    if strength.is_nan() {
        0.0
    } else {
        strength.clamp(0.0, PhraseBooster::MAX_STRENGTH)
    }
}

/// Case-insensitive, whitespace-normalised identity of a phrase.
fn dedup_key(phrase: &str) -> String {
    phrase
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// FNV-1a 64 over the piece strings joined by `\n`.
fn fingerprint(id_to_token: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &byte in bytes {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (i, piece) in id_to_token.iter().enumerate() {
        if i > 0 {
            eat(b"\n");
        }
        eat(piece.as_bytes());
    }
    h
}

/// One decode's walk through a booster. The decoder calls `pick` on each
/// joint output and `emitted` for each non-blank token it keeps. The tree
/// state moves only in `emitted`, so blanks, duration skips and forced
/// blanks leave it where it is.
pub(crate) struct BoostCursor<'a> {
    booster: &'a PhraseBooster,
    state: NodeId,
    /// Whether the latest `pick` differs from the unbiased argmax.
    last_changed: bool,
    changed: Vec<bool>,
    hits: Vec<RawHit>,
}

impl BoostCursor<'_> {
    /// The token to emit for these raw joint vocab logits (blank included).
    /// Blank versus non-blank is decided on the unbiased logits (research
    /// notes 1.4); boosting only re-ranks which non-blank token wins, by adding
    /// each token's bonus, already scaled by its phrases' strengths (notes 1.6).
    /// A token must strictly beat the unbiased pick's boosted score, so a
    /// booster with nothing to boost is plain argmax, ties included.
    pub(crate) fn pick(&mut self, vocab_logits: &[f32]) -> usize {
        let b = self.booster;
        let unbiased = argmax_last(vocab_logits).unwrap_or(b.blank_id);
        self.last_changed = false;
        if unbiased == b.blank_id {
            return unbiased;
        }
        let bonus = b.tree.step_bonus(self.state);
        // Identity by construction, not by arithmetic: with a NaN in the
        // logits, `unbiased` need not be the largest finite logit.
        if bonus.is_empty() {
            return unbiased;
        }
        let own = bonus
            .iter()
            .find(|&&(tok, _)| tok as usize == unbiased)
            .map_or(0.0, |&(_, x)| x);
        let mut best = unbiased;
        let mut best_score = vocab_logits[unbiased] + own;
        for &(tok, x) in bonus.iter() {
            let tok = tok as usize;
            if tok == b.blank_id {
                continue;
            }
            if let Some(&logit) = vocab_logits.get(tok) {
                let score = logit + x;
                if score > best_score {
                    best = tok;
                    best_score = score;
                }
            }
        }
        self.last_changed = best != unbiased;
        best
    }

    /// Records that `token` (non-blank) was emitted: whether the latest `pick`
    /// changed it, the new tree state, and a hit when a phrase completes here,
    /// spanning the last `depth` emitted tokens.
    pub(crate) fn emitted(&mut self, token: usize) {
        debug_assert_ne!(token, self.booster.blank_id, "blank is never emitted");
        self.changed.push(std::mem::take(&mut self.last_changed));
        self.state = self.booster.tree.advance(self.state, token as u32);
        if let Some((phrase, depth)) = self.booster.tree.completed(self.state) {
            let last = self.changed.len() - 1;
            self.hits.push(RawHit {
                phrase,
                first: last + 1 - usize::from(depth),
                last,
            });
        }
    }

    pub(crate) fn finish(self) -> BoostTrace {
        BoostTrace {
            changed: self.changed,
            hits: self.hits,
        }
    }
}

/// What one decode's cursor saw, indexed by emitted token.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct BoostTrace {
    /// Per emitted token: did boosting change it?
    pub changed: Vec<bool>,
    pub hits: Vec<RawHit>,
}

/// A completed phrase: `phrase` indexes `PhraseBooster::phrases()`, and
/// `first..=last` are emitted-token indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RawHit {
    pub phrase: u32,
    pub first: usize,
    pub last: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFF: usize = 3; // "▁the": in no phrase
    const CA: usize = 1; // "▁ca": first piece of "cairn"
    const IRN: usize = 2; // "irn": second piece of "cairn"
    const CAP_CA: usize = 7; // "▁Ca": first piece of "Cairn"

    fn vocab() -> (Vec<String>, usize) {
        let v: Vec<String> = [
            "<unk>", "▁ca", "irn", "▁the", "▁cat", "s", "▁dog", "▁Ca", "<blk>",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let blank = v.len() - 1;
        (v, blank)
    }

    /// Every phrase at `strength`.
    fn phrases(p: &[&str], strength: f32) -> Vec<BoostPhrase> {
        p.iter().map(|s| BoostPhrase::new(*s, strength)).collect()
    }

    fn texts(b: &PhraseBooster) -> Vec<&str> {
        b.phrases().iter().map(|p| p.text.as_str()).collect()
    }

    #[test]
    fn a_vocab_file_builds_the_same_booster_as_the_model_does() {
        let dir = std::env::temp_dir().join(format!("pb-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.txt");
        let tok_path = dir.join("tokenizer.json");
        let (v, blank) = vocab();
        let text: String = v.iter().enumerate().map(|(i, t)| format!("{t} {i}
")).collect();
        std::fs::write(&path, text).unwrap();
        spell::fixture::tokenizer(&v, blank).save(&tok_path, false).unwrap();
        let p = phrases(&["cairn", "Cairn", "ai"], 2.0);
        let from_file = PhraseBooster::from_files(&path, &tok_path, &p).unwrap();
        let direct = booster_for(&v, blank, &p);
        assert_eq!(from_file.vocab_fingerprint, direct.vocab_fingerprint);
        assert_eq!(from_file.vocab_len, direct.vocab_len);
        assert_eq!(from_file.blank_id, direct.blank_id, "blank is vocab size - 1 either way");
        assert_eq!(from_file.phrases(), direct.phrases());
        assert_eq!(from_file.spellings("cairn"), direct.spellings("cairn"));
        assert_eq!(from_file.root_fan_out(), direct.root_fan_out());
        assert!(from_file.check_vocab(&v).is_ok());
        assert!(PhraseBooster::from_files(&dir.join("missing.txt"), &tok_path, &p).is_err());
        assert!(PhraseBooster::from_files(&path, &dir.join("missing.json"), &p).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_phrases_need_no_tokenizer_file() {
        let dir = std::env::temp_dir().join(format!("pb-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.txt");
        let (v, blank) = vocab();
        let text: String = v.iter().enumerate().map(|(i, t)| format!("{t} {i}
")).collect();
        std::fs::write(&path, text).unwrap();
        let b = PhraseBooster::from_files(&path, &dir.join("no-such-tokenizer.json"), &[])
            .unwrap();
        assert!(b.is_empty());
        assert_eq!(b.blank_id, blank);
        assert!(b.check_vocab(&v).is_ok());
        // Its cursor is the real one: plain argmax, a trace per emitted token.
        let mut c = b.cursor();
        assert_eq!(c.pick(&logits(&[(OFF, 1.0)])), OFF);
        c.emitted(OFF);
        assert_eq!(c.finish().changed, vec![false]);
        // A non-empty list still needs the file.
        let p = phrases(&["cairn"], 2.0);
        assert!(PhraseBooster::from_files(&path, &dir.join("no-such-tokenizer.json"), &p)
            .is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pieces_are_the_vocabulary_by_id() {
        let b = booster(&["cairn"], 1.0);
        assert_eq!(b.pieces(), vocab().0.as_slice());
    }

    /// Every phrase at `strength`.
    fn booster(p: &[&str], strength: f32) -> PhraseBooster {
        let (v, b) = vocab();
        booster_for(&v, b, &phrases(p, strength))
    }

    /// A booster for `v` (blank `blank`), spelled by a fixture tokenizer for `v`.
    fn booster_for(v: &[String], blank: usize, p: &[BoostPhrase]) -> PhraseBooster {
        PhraseBooster::new(v, blank, spell::fixture::tokenizer(v, blank), p).unwrap()
    }

    /// Logits that are 0 everywhere except the listed tokens.
    fn logits(set: &[(usize, f32)]) -> Vec<f32> {
        let (v, _) = vocab();
        let mut l = vec![0.0; v.len()];
        for &(t, x) in set {
            l[t] = x;
        }
        l
    }

    /// Deterministic pseudo-random logits in [-4, 4).
    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn an_unbiased_blank_stays_blank_at_any_strength() {
        let (_, blank) = vocab();
        for strength in [0.0, 1.0, 5.0] {
            let b = booster(&["cairn"], strength);
            let mut c = b.cursor();
            // blank barely ahead of a phrase-first token at the root
            assert_eq!(c.pick(&logits(&[(blank, 1.0), (CA, 0.99)])), blank);
            // and of the continuation, mid-phrase, where the bonus is largest
            c.pick(&logits(&[(CA, 1.0)]));
            c.emitted(CA);
            assert_eq!(c.pick(&logits(&[(blank, 1.0), (IRN, 0.99)])), blank);
            let t = c.finish();
            assert_eq!(t.changed, vec![false]);
        }
    }

    #[test]
    fn an_unbiased_token_never_becomes_blank() {
        let (_, blank) = vocab();
        for strength in [0.0, 1.0, 5.0] {
            let b = booster(&["cairn"], strength);
            let mut c = b.cursor();
            // blank a hair behind an off-tree token: boosting may move the
            // pick to another non-blank token, never to blank
            let got = c.pick(&logits(&[(OFF, 1.0), (blank, 0.99), (CA, 0.5)]));
            assert_ne!(got, blank);
            c.emitted(CA);
            let got = c.pick(&logits(&[(OFF, 1.0), (blank, 0.99), (IRN, -3.0)]));
            assert_ne!(got, blank);
        }
    }

    #[test]
    fn zero_strength_is_plain_argmax_including_ties() {
        let b = booster(&["cairn", "dogs"], 0.0);
        let (v, blank) = vocab();
        let mut c = b.cursor();
        let mut cases: Vec<Vec<f32>> = (0..200).map(|s| noise(s, v.len())).collect();
        // exact ties between an off-tree token and a boosted one, both orders
        cases.push(logits(&[(CA, 2.0), (OFF, 2.0)]));
        cases.push(logits(&[(OFF, 2.0), (CAP_CA, 2.0)]));
        cases.push(logits(&[(IRN, 2.0), (OFF, 2.0)]));
        // a NaN makes argmax_last return a non-maximal later index; strength 0
        // must still follow it rather than re-rank against a boosted token
        cases.push(logits(&[(CA, 1.0), (2, f32::NAN), (OFF, 0.5)]));
        for l in &cases {
            let want = argmax_last(l).unwrap_or(blank);
            assert_eq!(c.pick(l), want, "{l:?}");
            if want != blank {
                c.emitted(want);
            }
        }
        assert!(c.finish().changed.iter().all(|&x| !x));
    }

    #[test]
    fn a_phrase_start_wins_a_near_tie_and_loses_a_clear_one() {
        // c0 = 1, so a phrase-first token is worth α nats at the root
        let l = logits(&[(OFF, 2.5), (CA, 2.0)]);
        assert_eq!(booster(&["cairn"], 1.0).cursor().pick(&l), CA);
        assert_eq!(booster(&["cairn"], 0.4).cursor().pick(&l), OFF);
    }

    #[test]
    fn the_state_moves_only_on_emitted() {
        let b = booster(&["cairn"], 1.0);
        let mut c = b.cursor();
        // `irn` is not a phrase start: at the root it gets nothing, however
        // often the cursor picks the phrase start without emitting it
        let mid = logits(&[(OFF, 3.0), (IRN, 1.0)]);
        for _ in 0..3 {
            assert_eq!(c.pick(&logits(&[(CA, 1.0)])), CA);
            assert_eq!(c.pick(&mid), OFF);
        }
        // once `▁ca` is emitted, `irn` continues the phrase: worth
        // node_score + token_score = 1 + (2 + ln 2) = 3.69 > 2
        c.emitted(CA);
        assert_eq!(c.pick(&mid), IRN);
        assert_eq!(c.pick(&mid), IRN);
    }

    #[test]
    fn a_completed_phrase_is_reported_with_its_span_and_changed_flag() {
        let b = booster(&["dogs", "cairn"], 1.0);
        let mut c = b.cursor();
        assert_eq!(c.pick(&logits(&[(OFF, 1.0)])), OFF);
        c.emitted(OFF);
        assert_eq!(c.pick(&logits(&[(OFF, 2.5), (CA, 2.0)])), CA); // changed by boosting
        c.emitted(CA);
        assert_eq!(c.pick(&logits(&[(IRN, 1.0)])), IRN); // unbiased already
        c.emitted(IRN);
        let t = c.finish();
        assert_eq!(t.changed, vec![false, true, false]);
        assert_eq!(
            t.hits,
            vec![RawHit {
                phrase: 1,
                first: 1,
                last: 2
            }]
        );
        assert_eq!(b.phrases()[1].text, "cairn");
    }

    #[test]
    fn duplicate_phrases_build_one_entry() {
        let b = booster(
            &["Cairn", "cairn", "Cairn", "  CAIRN ", "dogs", "dogs"],
            1.0,
        );
        assert_eq!(texts(&b), ["Cairn", "dogs"]);
        assert!(b.skipped().is_empty());
        // `▁ca`, `▁Ca`, `▁dog`: each phrase start once
        assert_eq!(b.root_fan_out(), 3);
        let want = vec![
            vec!["▁ca".to_string(), "irn".to_string()],
            vec!["▁Ca".to_string(), "irn".to_string()],
        ];
        assert_eq!(b.spellings("Cairn"), want);
        assert_eq!(b.spellings("cairn"), want);
        assert_eq!(
            b.spellings("dogs"),
            vec![vec!["▁dog".to_string(), "s".to_string()]]
        );
        assert!(b.spellings("unknown").is_empty());
    }

    #[test]
    fn case_duplicates_keep_every_variants_spellings() {
        let v: Vec<String> = ["<unk>", "▁dock", "box", "▁Dock", "Box", "<blk>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let blank = v.len() - 1;
        let p = |s: &str| s.to_string();
        let both = phrases(&["dockbox", "DockBox"], 1.0);
        let b = booster_for(&v, blank, &both);
        assert_eq!(texts(&b), ["dockbox"]);
        // `▁Dock Box` is only reachable through the second casing
        let want = vec![
            vec![p("▁dock"), p("box")],
            vec![p("▁Dock"), p("box")],
            vec![p("▁Dock"), p("Box")],
        ];
        assert_eq!(b.spellings("dockbox"), want);
        assert_eq!(b.spellings("DOCKBOX"), want);
        let mut c = b.cursor();
        c.emitted(3);
        c.emitted(4);
        assert_eq!(
            c.finish().hits,
            vec![RawHit {
                phrase: 0,
                first: 0,
                last: 1
            }]
        );
        // a later casing the vocab can spell rescues a first casing it cannot
        let caps: Vec<String> = ["<unk>", "▁Dock", "Box", "<blk>"]
            .iter()
            .map(|s| p(s))
            .collect();
        let b = booster_for(&caps, 3, &both);
        assert_eq!(texts(&b), ["dockbox"]);
        assert!(b.skipped().is_empty());
        assert_eq!(b.spellings("dockbox"), vec![vec![p("▁Dock"), p("Box")]]);
    }

    #[test]
    fn skipped_phrases_are_kept_with_their_reason() {
        let b = booster(&["AI", "cairn", "zebra", "   ", "zebra"], 1.0);
        assert_eq!(texts(&b), ["cairn"]);
        assert_eq!(
            b.skipped(),
            &[
                ("AI".to_string(), SkipReason::TooShort),
                ("zebra".to_string(), SkipReason::NotSpellable),
                ("   ".to_string(), SkipReason::TooShort),
            ]
        );
        assert!(!b.is_empty());
        assert!(booster(&["AI", "zebra"], 1.0).is_empty());
    }

    #[test]
    fn strength_is_clamped_and_zero_is_skipped() {
        assert_eq!(booster(&["cairn"], 9.0).phrases()[0].strength, 5.0);
        assert_eq!(booster(&["cairn"], 2.5).phrases()[0].strength, 2.5);
        for nothing in [0.0, -1.0, f32::NAN] {
            let b = booster(&["cairn"], nothing);
            assert!(b.is_empty(), "{nothing}");
            assert_eq!(b.skipped(), &[("cairn".to_string(), SkipReason::NoStrength)]);
        }
        // a spelling problem is the reason reported, whatever the strength
        assert_eq!(booster(&["AI"], 0.0).skipped(), &[("AI".to_string(), SkipReason::TooShort)]);
        assert_eq!(PhraseBooster::MAX_STRENGTH, 5.0);
    }

    #[test]
    fn each_phrase_is_boosted_at_its_own_strength() {
        // `▁ca` (cairn, 1.0) and `▁dog` (dogs, 0.4) each 0.5 behind an off-tree
        // token at the root: only the stronger phrase's start overtakes it.
        let (v, blank) = vocab();
        let p = [BoostPhrase::new("cairn", 1.0), BoostPhrase::new("dogs", 0.4)];
        let b = booster_for(&v, blank, &p);
        assert_eq!(b.cursor().pick(&logits(&[(OFF, 2.5), (CA, 2.0)])), CA);
        assert_eq!(b.cursor().pick(&logits(&[(OFF, 2.5), (6, 2.0)])), OFF);
        assert_eq!(b.phrases(), &p);
    }

    #[test]
    fn case_duplicates_take_the_larger_strength() {
        let (v, blank) = vocab();
        let p = [BoostPhrase::new("cairn", 0.5), BoostPhrase::new("Cairn", 1.0)];
        let b = booster_for(&v, blank, &p);
        assert_eq!(b.phrases(), &[BoostPhrase::new("cairn", 1.0)]);
    }

    #[test]
    fn a_booster_for_another_vocabulary_is_refused() {
        let (v, _) = vocab();
        let b = booster(&["cairn"], 1.0);
        assert!(b.check_vocab(&v).is_ok());
        let mut renamed = v.clone();
        renamed[4] = "▁cot".to_string(); // same length, one piece differs
        assert!(matches!(
            b.check_vocab(&renamed),
            Err(crate::error::Error::Config(_))
        ));
        assert!(b.check_vocab(&v[..v.len() - 1]).is_err());
    }

    #[test]
    fn an_empty_booster_still_runs_its_cursor() {
        let (v, blank) = vocab();
        let b = booster(&[], 5.0);
        assert!(b.is_empty());
        assert_eq!(b.root_fan_out(), 0);
        let mut c = b.cursor();
        let mut emitted = 0;
        for s in 0..200 {
            let l = noise(s, v.len());
            let want = argmax_last(&l).unwrap_or(blank);
            assert_eq!(c.pick(&l), want);
            if want != blank {
                c.emitted(want);
                emitted += 1;
            }
        }
        let t = c.finish();
        assert_eq!(t.changed.len(), emitted);
        assert!(t.changed.iter().all(|&x| !x));
        assert!(t.hits.is_empty());
    }
}
