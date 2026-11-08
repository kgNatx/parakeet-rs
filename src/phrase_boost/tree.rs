//! The boosting tree: an Aho-Corasick automaton over phrase token sequences,
//! scored like NeMo's boosting tree (research notes 1.1-1.3).
//!
//! Knows: node scores, fail links, backoff, the score of any token from any
//! state, and which phrase a state has just completed. Each sequence carries a
//! weight (its phrase's boost strength) that scales the score of every token on
//! its path, so a step bonus is already in nats. Does not know about the decoder
//! or the vocabulary.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};

pub(crate) type NodeId = u32;
pub(crate) const ROOT: NodeId = 0;

/// Built once from the phrases' token sequences; immutable afterwards.
///
/// `build` takes `(token sequence, phrase index, weight)` triples. Every query
/// is a pure function of a `NodeId` state, so a decode needs only one `NodeId`.
/// Scores follow NeMo, times a weight: the token at 0-indexed depth `i` is worth
/// `w * c0` when `i == 0` and `w * (c0 * beta + ln(i + 1))` otherwise, where `w`
/// is the node's weight; a node's score is the sum along its path, and tokens
/// with no arc on the fail chain score 0.
///
/// A node's weight is the LARGEST weight of the sequences through it, so a
/// prefix shared by a strong and a weak phrase is boosted as the strong one
/// until they diverge. With every weight 1.0 the arithmetic is NeMo's exactly
/// (`x * 1.0 == x`), and with every weight `w` each score is `w` times that.
pub(crate) struct Tree {
    nodes: Vec<Node>,
    c0: f32,
    beta: f32,
    /// `step_bonus(ROOT)`, computed once: it is asked at every decoder step
    /// that is not mid-phrase.
    root_bonus: Vec<(u32, f32)>,
}

struct Node {
    children: BTreeMap<u32, NodeId>,
    fail: NodeId,
    depth: u16,
    /// The largest weight of the sequences through this node; 0 at the root.
    weight: f32,
    score: f32,
    /// Phrase index when a phrase ends exactly here.
    end: Option<u32>,
    /// Nearest end node on the chain `self, fail, fail(fail), ...`.
    output: Option<NodeId>,
}

fn token_score(c0: f32, beta: f32, i: u16) -> f32 {
    if i == 0 {
        c0
    } else {
        c0 * beta + (f32::from(i) + 1.0).ln()
    }
}

impl Tree {
    /// Empty sequences are ignored. When two sequences are identical, the
    /// first one's phrase index is kept (and the larger weight). Weights must
    /// be finite and non-negative; the booster clamps them before this.
    pub(crate) fn build(seqs: &[(Vec<u32>, u32, f32)], c0: f32, beta: f32) -> Tree {
        let new_node = |depth: u16| Node {
            children: BTreeMap::new(),
            fail: ROOT,
            depth,
            weight: 0.0,
            score: 0.0,
            end: None,
            output: None,
        };
        let mut nodes = vec![new_node(0)];
        for (seq, phrase, weight) in seqs {
            debug_assert!(weight.is_finite() && *weight >= 0.0, "weight {weight}");
            if seq.is_empty() {
                continue;
            }
            let mut cur = ROOT;
            for &tok in seq {
                cur = match nodes[cur as usize].children.get(&tok) {
                    Some(&next) => next,
                    None => {
                        let depth = nodes[cur as usize].depth + 1;
                        let id = nodes.len() as NodeId;
                        nodes.push(new_node(depth));
                        nodes[cur as usize].children.insert(tok, id);
                        id
                    }
                };
                let w = &mut nodes[cur as usize].weight;
                *w = w.max(*weight);
            }
            nodes[cur as usize].end.get_or_insert(*phrase);
        }

        // Breadth-first: a node's fail target is always shallower, so already final.
        // `order` is the BFS order of every non-root node.
        let mut order: Vec<NodeId> = Vec::with_capacity(nodes.len());
        let mut queue: VecDeque<NodeId> = nodes[ROOT as usize].children.values().copied().collect();
        for &first in &queue {
            nodes[first as usize].score = nodes[first as usize].weight * token_score(c0, beta, 0);
        }
        while let Some(parent) = queue.pop_front() {
            order.push(parent);
            let kids: Vec<(u32, NodeId)> =
                nodes[parent as usize].children.iter().map(|(&t, &c)| (t, c)).collect();
            for (tok, child) in kids {
                let mut f = nodes[parent as usize].fail;
                let fail = loop {
                    match nodes[f as usize].children.get(&tok) {
                        Some(&hit) if hit != child => break hit,
                        _ if f == ROOT => break ROOT,
                        _ => f = nodes[f as usize].fail,
                    }
                };
                nodes[child as usize].fail = fail;
                // A parent's score is final before its children's (breadth-first).
                let parent_score = nodes[parent as usize].score;
                let arc = nodes[child as usize].weight
                    * token_score(c0, beta, nodes[parent as usize].depth);
                nodes[child as usize].score = parent_score + arc;
                queue.push_back(child);
            }
        }
        for &n in &order {
            let fail = nodes[n as usize].fail;
            nodes[n as usize].output =
                if nodes[n as usize].end.is_some() { Some(n) } else { nodes[fail as usize].output };
        }

        let mut tree = Tree { nodes, c0, beta, root_bonus: Vec::new() };
        tree.root_bonus = tree.compute_bonus(ROOT);
        tree
    }

    /// The Aho-Corasick goto: follow fail links until `tok` has an arc. At the
    /// root with no arc, stay at the root.
    pub(crate) fn advance(&self, s: NodeId, tok: u32) -> NodeId {
        let mut cur = s;
        loop {
            if let Some(&next) = self.nodes[cur as usize].children.get(&tok) {
                return next;
            }
            if cur == ROOT {
                return ROOT;
            }
            cur = self.nodes[cur as usize].fail;
        }
    }

    /// The absolute score of emitting `tok` from state `s`: the backoffs paid
    /// on the way to the first node with an arc for `tok`, plus that arc's
    /// score. A token with no arc anywhere on the chain pays all backoffs and
    /// gains 0. The decoder uses `step_bonus`; this is the definition the tests
    /// check it against.
    #[cfg(test)]
    pub(crate) fn score(&self, s: NodeId, tok: u32) -> f32 {
        let mut acc = 0.0;
        let mut cur = s;
        loop {
            if let Some(&child) = self.nodes[cur as usize].children.get(&tok) {
                return acc + self.arc_score(cur, child);
            }
            if cur == ROOT {
                return acc;
            }
            acc += self.backoff(cur);
            cur = self.nodes[cur as usize].fail;
        }
    }

    /// `score(s, tok) - score(s, <any token with no arc>)` for every token
    /// with an arc on `s`'s fail chain; every other token's bonus is 0. Adding
    /// a constant to every token cannot change an argmax, so this is all a
    /// greedy decoder needs. Each token is listed once, in chain order.
    pub(crate) fn step_bonus(&self, s: NodeId) -> Cow<'_, [(u32, f32)]> {
        if s == ROOT {
            Cow::Borrowed(&self.root_bonus)
        } else {
            Cow::Owned(self.compute_bonus(s))
        }
    }

    /// The phrase and end depth of the nearest phrase ending at `s` or at a
    /// suffix of `s`'s path.
    pub(crate) fn completed(&self, s: NodeId) -> Option<(u32, u16)> {
        let node = &self.nodes[self.nodes[s as usize].output? as usize];
        node.end.map(|phrase| (phrase, node.depth))
    }

    /// Number of distinct phrase-first tokens.
    pub(crate) fn root_fan_out(&self) -> usize {
        self.nodes[ROOT as usize].children.len()
    }

    #[cfg(test)]
    pub(crate) fn node_score(&self, s: NodeId) -> f32 {
        self.nodes[s as usize].score
    }

    #[cfg(test)]
    pub(crate) fn fail(&self, s: NodeId) -> NodeId {
        self.nodes[s as usize].fail
    }

    /// The node reached by `path` from the root, with no fail links. Panics if
    /// the path is not in the tree; this is a test accessor.
    #[cfg(test)]
    pub(crate) fn node_for(&self, path: &[u32]) -> NodeId {
        path.iter().fold(ROOT, |cur, tok| {
            *self.nodes[cur as usize]
                .children
                .get(tok)
                .unwrap_or_else(|| panic!("path {path:?} is not in the tree"))
        })
    }

    /// Score of the arc `parent -> child`: the child's weight times the score
    /// of a token at the parent's depth (the token's 0-indexed position).
    /// Equal to `score(child) - score(parent)`, computed as `build` does.
    fn arc_score(&self, parent: NodeId, child: NodeId) -> f32 {
        self.nodes[child as usize].weight
            * token_score(self.c0, self.beta, self.nodes[parent as usize].depth)
    }

    /// Zero at the root and at end nodes (a finished phrase keeps its bonus);
    /// otherwise the score lost by falling back to the fail node.
    fn backoff(&self, s: NodeId) -> f32 {
        let n = &self.nodes[s as usize];
        if s == ROOT || n.end.is_some() {
            0.0
        } else {
            self.nodes[n.fail as usize].score - n.score
        }
    }

    fn compute_bonus(&self, s: NodeId) -> Vec<(u32, f32)> {
        let mut chain = vec![s];
        while *chain.last().unwrap() != ROOT {
            chain.push(self.nodes[*chain.last().unwrap() as usize].fail);
        }
        let base: f32 = chain.iter().map(|&n| self.backoff(n)).sum();
        let mut out: Vec<(u32, f32)> = Vec::new();
        let mut acc = 0.0;
        for &n in &chain {
            for (&tok, &child) in &self.nodes[n as usize].children {
                if !out.iter().any(|(t, _)| *t == tok) {
                    out.push((tok, acc + self.arc_score(n, child) - base));
                }
            }
            acc += self.backoff(n);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn fixture() -> Tree {
        Tree::build(&[(vec![1, 2, 3], 0, 1.0), (vec![1, 2, 4], 1, 1.0), (vec![3], 2, 1.0)], 1.0, 1.0)
    }

    /// Per-token boost along a decoded sequence, following the automaton.
    fn run(t: &Tree, toks: &[u32]) -> Vec<f32> {
        let mut s = ROOT;
        toks.iter()
            .map(|&v| {
                let sc = t.score(s, v);
                s = t.advance(s, v);
                sc
            })
            .collect()
    }

    fn assert_vec(got: Vec<f32>, want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(close(*g, *w), "got {got:?} want {want:?}");
        }
    }

    #[test]
    fn nemo_fixture_node_scores_and_fail_links() {
        let t = fixture();
        let abc = t.node_for(&[1, 2, 3]);
        let abd = t.node_for(&[1, 2, 4]);
        let c = t.node_for(&[3]);
        assert!(close(t.node_score(abc), 4.7918));
        assert!(close(t.node_score(abd), 4.7918));
        assert!(close(t.node_score(c), 1.0));
        assert_eq!(t.fail(abc), c);
        assert_eq!(t.fail(abd), ROOT);
    }

    #[test]
    fn nemo_fixture_per_token_scores() {
        let t = fixture();
        assert_vec(run(&t, &[1, 2, 3, 2, 1]), &[1.0, 1.6931, 2.0986, 0.0, 1.0]);
        assert_vec(run(&t, &[2, 2, 1, 2, 4]), &[0.0, 0.0, 1.0, 1.6931, 2.0986]);
        assert_vec(run(&t, &[3, 1, 2, 1]), &[1.0, 1.0, 1.6931, -1.6931]);
    }

    #[test]
    fn default_depth_scores() {
        let t = Tree::build(&[(vec![1, 2, 3, 4, 5], 0, 1.0)], 1.0, 2.0);
        let want = [1.0, 3.6931, 6.7918, 10.1781, 13.7875];
        for (i, w) in want.iter().enumerate() {
            let n = t.node_for(&[1, 2, 3, 4, 5][..=i]);
            assert!(close(t.node_score(n), *w), "depth {} {}", i + 1, t.node_score(n));
        }
    }

    #[test]
    fn depth_scores_scale_with_c0() {
        // c0 = 2: the first token is worth c0, every later one c0 * beta + ln(i + 1)
        let t = Tree::build(&[(vec![1, 2, 3], 0, 1.0)], 2.0, 2.0);
        let want = [2.0, 2.0 + 4.0 + 2f32.ln(), 2.0 + 8.0 + 2f32.ln() + 3f32.ln()];
        for (i, w) in want.iter().enumerate() {
            let n = t.node_for(&[1, 2, 3][..=i]);
            assert!(close(t.node_score(n), *w), "depth {} {}", i + 1, t.node_score(n));
        }
        assert!(close(t.score(ROOT, 1), 2.0));
    }

    #[test]
    fn relative_bonus_equals_brute_force() {
        let t = Tree::build(
            &[(vec![1, 2, 3], 0, 1.0), (vec![1, 2, 4], 1, 1.0), (vec![3], 2, 1.0), (vec![2, 3, 1], 3, 1.0)],
            1.0,
            2.0,
        );
        for s in 0..t.nodes.len() as NodeId {
            let base = t.score(s, 5); // 5 is off-tree
            let bonus = t.step_bonus(s);
            for v in 0..6u32 {
                let b = bonus.iter().find(|(tok, _)| *tok == v).map_or(0.0, |(_, x)| *x);
                assert!(close(b, t.score(s, v) - base), "s={s} v={v} b={b}");
            }
            // each token listed once
            let mut toks: Vec<u32> = bonus.iter().map(|(tok, _)| *tok).collect();
            toks.sort();
            toks.dedup();
            assert_eq!(toks.len(), bonus.len());
        }
    }

    #[test]
    fn advance_is_aho_corasick_goto() {
        let t = fixture();
        let ab = t.node_for(&[1, 2]);
        assert_eq!(t.advance(ab, 3), t.node_for(&[1, 2, 3]));
        assert_ne!(t.advance(ab, 3), t.node_for(&[3]));
        assert_eq!(t.advance(ab, 5), ROOT);
        assert_eq!(t.advance(ROOT, 5), ROOT);
        // from `ab`, token 1 falls back to the root arc for `a`
        assert_eq!(t.advance(ab, 1), t.node_for(&[1]));
    }

    #[test]
    fn completed_reports_suffix_phrases() {
        let t = Tree::build(&[(vec![1, 2, 3, 4], 0, 1.0), (vec![2, 3], 1, 1.0)], 1.0, 2.0);
        let mut s = ROOT;
        let mut last = None;
        for v in [1, 2, 3] {
            s = t.advance(s, v);
            last = t.completed(s);
        }
        assert_eq!(last, Some((1, 2)));
        assert_eq!(t.completed(t.node_for(&[1, 2])), None);
        assert_eq!(t.completed(t.node_for(&[1, 2, 3, 4])), Some((0, 4)));
        assert_eq!(t.root_fan_out(), 2);
    }

    #[test]
    fn duplicate_sequences_keep_the_first_phrase() {
        let t = Tree::build(&[(vec![1, 2], 7, 1.0), (vec![1, 2], 9, 1.0)], 1.0, 2.0);
        assert_eq!(t.completed(t.node_for(&[1, 2])), Some((7, 2)));
    }

    #[test]
    fn a_uniform_weight_scales_every_bonus_exactly() {
        // 0.5 is a power of two, so scaling is exact: the booster at 0.5 for
        // every phrase must equal the old global strength 0.5 bit for bit.
        let seqs = |w: f32| -> Vec<(Vec<u32>, u32, f32)> {
            vec![(vec![1, 2, 3], 0, w), (vec![1, 2, 4], 1, w), (vec![3], 2, w), (vec![2, 3, 1], 3, w)]
        };
        let one = Tree::build(&seqs(1.0), 1.0, 2.0);
        let half = Tree::build(&seqs(0.5), 1.0, 2.0);
        assert_eq!(one.nodes.len(), half.nodes.len());
        for s in 0..one.nodes.len() as NodeId {
            let (a, b) = (one.step_bonus(s), half.step_bonus(s));
            assert_eq!(a.len(), b.len());
            for (&(ta, xa), &(tb, xb)) in a.iter().zip(b.iter()) {
                assert_eq!(ta, tb);
                assert_eq!((0.5 * xa).to_bits(), xb.to_bits(), "s={s} tok={ta}");
            }
        }
    }

    #[test]
    fn a_shared_prefix_takes_the_stronger_weight() {
        // 1 2 3 at 1.0 and 1 2 4 at 0.5 share `1 2`: boosted at 1.0 until they part.
        let t = Tree::build(&[(vec![1, 2, 3], 0, 1.0), (vec![1, 2, 4], 1, 0.5)], 1.0, 2.0);
        let full = Tree::build(&[(vec![1, 2, 3], 0, 1.0)], 1.0, 2.0);
        let ab = t.node_for(&[1, 2]);
        assert_eq!(t.node_score(ab).to_bits(), full.node_score(full.node_for(&[1, 2])).to_bits());
        let arc = |n: &[u32]| t.node_score(t.node_for(n)) - t.node_score(ab);
        assert!(close(arc(&[1, 2, 3]), 2.0 + 3f32.ln()));
        assert!(close(arc(&[1, 2, 4]), 0.5 * (2.0 + 3f32.ln())));
    }

    #[test]
    fn a_weak_phrase_alone_is_boosted_at_its_own_weight() {
        let t = Tree::build(&[(vec![1, 2], 0, 1.0), (vec![3, 4], 1, 0.5)], 1.0, 2.0);
        assert!(close(t.score(ROOT, 1), 1.0));
        assert!(close(t.score(ROOT, 3), 0.5));
    }

    #[test]
    fn relative_bonus_equals_brute_force_with_mixed_weights() {
        let t = Tree::build(
            &[(vec![1, 2, 3], 0, 1.0), (vec![1, 2, 4], 1, 0.5), (vec![3], 2, 0.25), (vec![2, 3, 1], 3, 2.0)],
            1.0,
            2.0,
        );
        for s in 0..t.nodes.len() as NodeId {
            let base = t.score(s, 5);
            let bonus = t.step_bonus(s);
            for v in 0..6u32 {
                let b = bonus.iter().find(|(tok, _)| *tok == v).map_or(0.0, |(_, x)| *x);
                assert!(close(b, t.score(s, v) - base), "s={s} v={v} b={b}");
            }
        }
    }
}
