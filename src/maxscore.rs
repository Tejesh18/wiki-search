//! Block-max MaxScore (Turtle & Flood 1995, block maxima per Ding & Suel 2011).
//!
//! Term-at-a-time scoring touches every document that has any query term,
//! which for "who sang take me home country roads" is most of Wikipedia.
//! Here documents are visited in id order, and two kinds of upper bounds let
//! most of them be skipped:
//!
//! - per term: the best score that term gives any document. Once the k-th
//!   best score so far beats the sum of the weakest terms' bounds, a document
//!   that only has those terms can't make it, so their lists stop driving the
//!   loop ("non-essential") and are only probed for candidates.
//! - per block of 128 postings: the same bound for just that block, which is
//!   much tighter, used to drop a candidate before decoding anything.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::bm25::{Hit, Searcher};
use crate::index::{Index, TermInfo};
use crate::varint;

pub const BLOCK: usize = 128;
const END: u32 = u32::MAX;

/// Skip data for one term's postings, built when the index is opened.
pub struct Blocks {
    start: Vec<u32>, // byte offset of each block in the term's postings
    base: Vec<u32>,  // doc id just before the block (gaps continue across blocks)
    last: Vec<u32>,  // last doc id in the block
    max_w: Vec<f32>, // best tf/length weight in the block, before idf
    pub max: f32,    // best over the whole list
}

pub fn blocks_for(index: &Index, info: TermInfo, norm: &[f32], k1: f32) -> Blocks {
    let mut b = Blocks { start: vec![], base: vec![], last: vec![], max_w: vec![], max: 0.0 };
    let bytes = index.postings_of(info);
    let (mut pos, mut doc, mut n) = (0usize, 0u32, 0usize);
    while pos < bytes.len() {
        if n % BLOCK == 0 {
            b.start.push(pos as u32);
            b.base.push(doc);
            b.last.push(0);
            b.max_w.push(0.0);
        }
        doc += varint::read(bytes, &mut pos);
        let tf = varint::read(bytes, &mut pos) as f32;
        let w = tf * (k1 + 1.0) / (tf + norm[doc as usize]);
        let i = b.last.len() - 1;
        b.last[i] = doc;
        b.max_w[i] = b.max_w[i].max(w);
        b.max = b.max.max(w);
        n += 1;
    }
    b
}

struct Cursor<'a> {
    bytes: &'a [u8],
    blocks: &'a Blocks,
    pos: usize,
    n: usize, // postings decoded so far
    doc: u32,
    tf: u32,
    idf: f32,
    ub: f32,
    shallow: usize, // block used for bounds; only moves forward
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], blocks: &'a Blocks, idf: f32) -> Self {
        let mut c = Cursor { bytes, blocks, pos: 0, n: 0, doc: 0, tf: 0, idf, ub: idf * blocks.max, shallow: 0 };
        c.next();
        c
    }

    #[inline]
    fn next(&mut self) {
        if self.pos >= self.bytes.len() {
            self.doc = END;
            return;
        }
        self.doc += varint::read(self.bytes, &mut self.pos);
        self.tf = varint::read(self.bytes, &mut self.pos);
        self.n += 1;
    }

    /// Move to the first posting with doc >= target, jumping whole blocks.
    fn seek(&mut self, target: u32) {
        if self.doc >= target {
            return;
        }
        let b = self.blocks;
        let mut nb = (self.n - 1) / BLOCK;
        while nb < b.last.len() && b.last[nb] < target {
            nb += 1;
        }
        if nb == b.last.len() {
            self.doc = END;
            return;
        }
        if nb > (self.n - 1) / BLOCK {
            self.pos = b.start[nb] as usize;
            self.doc = b.base[nb];
            self.n = nb * BLOCK;
            self.next();
        }
        while self.doc < target {
            self.next();
        }
    }

    /// Upper bound on this term's score for `doc`, from its block, without decoding.
    #[inline]
    fn block_bound(&mut self, doc: u32) -> f32 {
        let b = self.blocks;
        while self.shallow < b.last.len() && b.last[self.shallow] < doc {
            self.shallow += 1;
        }
        if self.shallow == b.last.len() {
            0.0
        } else {
            self.idf * b.max_w[self.shallow]
        }
    }
}

impl<'a> Searcher<'a> {
    pub fn search_maxscore(&mut self, query: &str, k: usize) -> Vec<(u32, f32)> {
        let mut terms = Vec::new();
        self.tok.tokens(query, &mut terms);
        for t in &terms {
            self.ensure_blocks(t);
        }
        let index = self.index;
        let mut cur: Vec<Cursor> = terms
            .iter()
            .filter_map(|t| {
                let info = *index.terms.get(t)?;
                let blocks = self.blocks.get(t)?;
                Some(Cursor::new(index.postings_of(info), blocks, self.idf(info.df)))
            })
            .collect();
        if cur.is_empty() {
            return vec![];
        }
        cur.sort_by(|a, b| a.ub.total_cmp(&b.ub));
        // cum[i]: the most terms 0..=i can add to any document
        let cum: Vec<f32> = cur.iter().scan(0.0, |s, c| { *s += c.ub; Some(*s) }).collect();
        let (norm, k1) = (&self.norm, self.k1);
        let weight = |tf: u32, d: u32| { let tf = tf as f32; tf * (k1 + 1.0) / (tf + norm[d as usize]) };

        let mut heap: BinaryHeap<Reverse<Hit>> = BinaryHeap::with_capacity(k + 1);
        let mut theta = 0.0f32;
        let mut essential = 0; // cursors before this index are non-essential
        loop {
            let d = cur[essential..].iter().map(|c| c.doc).min().unwrap_or(END);
            if d == END {
                break;
            }
            let mut score = 0.0;
            for c in cur[essential..].iter_mut() {
                if c.doc == d {
                    score += c.idf * weight(c.tf, d);
                    c.next();
                }
            }
            if essential > 0 {
                // cheap check first: could the non-essential terms close the gap?
                let bound: f32 = cur[..essential].iter_mut().map(|c| c.block_bound(d)).sum();
                if score + bound > theta {
                    for i in (0..essential).rev() {
                        if score + cum[i] <= theta {
                            break;
                        }
                        let c = &mut cur[i];
                        c.seek(d);
                        if c.doc == d {
                            score += c.idf * weight(c.tf, d);
                        }
                    }
                }
            }
            if heap.len() < k {
                heap.push(Reverse(Hit(score, d)));
            } else if score > theta {
                heap.pop();
                heap.push(Reverse(Hit(score, d)));
            }
            if heap.len() == k {
                theta = heap.peek().unwrap().0 .0;
                while essential < cur.len() && cum[essential] <= theta {
                    essential += 1;
                }
            }
        }
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|Reverse(Hit(s, d))| (d, s)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}
