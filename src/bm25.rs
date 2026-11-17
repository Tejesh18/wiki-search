//! BM25, scored term at a time: one score slot per document, each query
//! term adds its contribution to the documents in its postings list, then
//! the top k slots are picked out with a small heap.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::index::Index;
use crate::maxscore::{blocks_for, Blocks};
use crate::tokenize::Tokenizer;

// k1: how fast repeated terms stop counting. b: how much long documents are
// penalized. Anserini's defaults; override with BM25_K1 / BM25_B.
fn param(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[derive(PartialEq, PartialOrd)]
pub struct Hit(pub f32, pub u32);
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct Searcher<'a> {
    pub(crate) index: &'a Index,
    pub(crate) tok: Tokenizer,
    scores: Vec<f32>,
    touched: Vec<u32>,
    // the length part of BM25's denominator, precomputed per document
    pub(crate) norm: Vec<f32>,
    pub(crate) k1: f32,
    // skip data for MaxScore, built the first time a term is queried
    pub(crate) blocks: HashMap<String, Blocks>,
}

impl<'a> Searcher<'a> {
    pub fn new(index: &'a Index) -> Self {
        let (k1, b) = (param("BM25_K1", 0.9), param("BM25_B", 0.4));
        let norm = index.doc_len.iter().map(|&l| k1 * (1.0 - b + b * l as f32 / index.avg_len)).collect();
        Searcher { index, tok: Tokenizer::new(), scores: vec![0.0; index.num_docs()], touched: Vec::new(), norm, k1, blocks: HashMap::new() }
    }

    pub(crate) fn ensure_blocks(&mut self, term: &str) {
        if self.blocks.contains_key(term) {
            return;
        }
        if let Some(&info) = self.index.terms.get(term) {
            let b = blocks_for(self.index, info, &self.norm, self.k1);
            self.blocks.insert(term.to_string(), b);
        }
    }

    pub fn idf(&self, df: u32) -> f32 {
        let n = self.index.num_docs() as f32;
        (1.0 + (n - df as f32 + 0.5) / (df as f32 + 0.5)).ln()
    }

    pub fn search(&mut self, query: &str, k: usize) -> Vec<(u32, f32)> {
        let mut terms = Vec::new();
        self.tok.tokens(query, &mut terms);
        for t in &terms {
            let Some(&info) = self.index.terms.get(t) else { continue };
            let idf = self.idf(info.df);
            let (scores, touched, norm, k1) = (&mut self.scores, &mut self.touched, &self.norm, self.k1);
            self.index.for_each_posting(info, |doc, tf| {
                let d = doc as usize;
                if scores[d] == 0.0 {
                    touched.push(doc);
                }
                let tf = tf as f32;
                scores[d] += idf * tf * (k1 + 1.0) / (tf + norm[d]);
            });
        }

        let mut heap: BinaryHeap<Reverse<Hit>> = BinaryHeap::with_capacity(k + 1);
        for &doc in &self.touched {
            let s = self.scores[doc as usize];
            if heap.len() < k {
                heap.push(Reverse(Hit(s, doc)));
            } else if s > heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push(Reverse(Hit(s, doc)));
            }
            self.scores[doc as usize] = 0.0;
        }
        self.touched.clear();
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|Reverse(Hit(s, d))| (d, s)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}
