//! Phrase boosting for the greedy TDT decoder.
//!
//! Knows how a list of phrases biases a decode: how a phrase is spelled in the
//! model's vocab pieces, the boosting tree, and the per-step token pick.

mod spell;
mod tree;

/// Why a phrase was left out of the boost list. Never fatal: the caller logs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Two letters or fewer: too ambiguous to boost.
    TooShort,
    /// The vocab cannot spell it (a character with no piece).
    NotSpellable,
}
