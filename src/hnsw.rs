//! HNSW (Malkov & Yashunin 2016): a layered proximity graph for approximate
//! nearest-neighbour search. Every vector is a node on layer 0; a shrinking
//! random subset also lives on higher layers, which act like express lanes.
//! A search starts at the top, walks greedily toward the query, and drops a
//! layer each time it can't get closer, then does a wider best-first search
//! on layer 0.
//!
//! Built with all cores: each insert only locks the neighbour list it is
//! changing, so threads can add nodes at the same time.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct D(f32, u32); // distance, node
impl Eq for D {}
impl Ord for D {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // eight accumulators so the compiler can keep a whole AVX register busy
    let mut acc = [0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

/// Vectors as one flat row-major array, scaled to unit length so that
/// cosine similarity is just a dot product.
pub struct Vectors {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Vectors {
    pub fn load(path: &Path, dim: usize) -> Result<Vectors> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let mut data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        for v in data.chunks_exact_mut(dim) {
            let n = dot(v, v).sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= n);
        }
        Ok(Vectors { dim, data })
    }
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    #[inline]
    pub fn get(&self, i: u32) -> &[f32] {
        &self.data[i as usize * self.dim..(i as usize + 1) * self.dim]
    }
}

/// Which nodes a search has already looked at. Generation-stamped so it
/// doesn't need clearing between searches.
pub struct Visited {
    mark: Vec<u32>,
    gen: u32,
}
impl Visited {
    pub fn new(n: usize) -> Self {
        Visited { mark: vec![0; n], gen: 0 }
    }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.gen = 1;
        }
    }
    #[inline]
    fn insert(&mut self, i: u32) -> bool {
        let m = &mut self.mark[i as usize];
        if *m == self.gen { false } else { *m = self.gen; true }
    }
}

pub struct Params {
    pub m: usize,      // links per node on upper layers
    pub ef_build: usize,
}

pub struct Hnsw {
    m: usize,
    m0: usize, // layer 0 gets twice as many links
    ef_build: usize,
    links: Vec<RwLock<Vec<Vec<u32>>>>, // links[node][layer]
    entry: Mutex<(u32, usize)>,       // entry node, top layer
}

impl Hnsw {
    fn dist(v: &Vectors, q: &[f32], i: u32) -> f32 {
        1.0 - dot(q, v.get(i))
    }

    /// Best-first search on one layer. Returns up to `ef` closest, nearest first.
    fn search_layer(&self, v: &Vectors, q: &[f32], entry: &[u32], ef: usize, layer: usize, seen: &mut Visited) -> Vec<D> {
        seen.reset();
        let mut cand = BinaryHeap::new(); // nearest on top
        let mut best = BinaryHeap::new(); // farthest on top
        for &e in entry {
            if seen.insert(e) {
                let d = D(Self::dist(v, q, e), e);
                cand.push(Reverse(d));
                best.push(d);
            }
        }
        while let Some(Reverse(c)) = cand.pop() {
            if best.len() >= ef && c.0 > best.peek().unwrap().0 {
                break; // nothing left that can improve the result
            }
            let nbrs = self.links[c.1 as usize].read()[layer].clone();
            for n in nbrs {
                if !seen.insert(n) {
                    continue;
                }
                let d = Self::dist(v, q, n);
                if best.len() < ef || d < best.peek().unwrap().0 {
                    cand.push(Reverse(D(d, n)));
                    best.push(D(d, n));
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        best.into_sorted_vec()
    }

    /// Keep a candidate only if it's closer to the base than to every neighbour
    /// already kept. Spreads links out in different directions instead of
    /// spending them all on one tight cluster, which is what keeps the graph
    /// navigable.
    fn select(v: &Vectors, cands: &[D], m: usize) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(m);
        for c in cands {
            if out.len() == m {
                break;
            }
            let x = v.get(c.1);
            if out.iter().all(|&o| 1.0 - dot(x, v.get(o)) > c.0) {
                out.push(c.1);
            }
        }
        out
    }

    fn greedy(&self, v: &Vectors, q: &[f32], top: usize, down_to: usize, start: u32) -> u32 {
        let (mut cur, mut cur_d) = (start, Self::dist(v, q, start));
        for layer in (down_to..=top).rev() {
            loop {
                let mut moved = false;
                let nbrs = self.links[cur as usize].read()[layer].clone();
                for n in nbrs {
                    let d = Self::dist(v, q, n);
                    if d < cur_d {
                        cur = n;
                        cur_d = d;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        cur
    }

    fn insert(&self, v: &Vectors, node: u32, level: usize, seen: &mut Visited) {
        let q = v.get(node);
        let (ep, top) = *self.entry.lock();
        let mut eps = vec![if level < top { self.greedy(v, q, top, level + 1, ep) } else { ep }];
        for layer in (0..=level.min(top)).rev() {
            let cands = self.search_layer(v, q, &eps, self.ef_build, layer, seen);
            let mine = Self::select(v, &cands, self.m);
            self.links[node as usize].write()[layer] = mine.clone();
            let cap = if layer == 0 { self.m0 } else { self.m };
            for &n in &mine {
                let mut l = self.links[n as usize].write();
                let list = &mut l[layer];
                list.push(node);
                if list.len() > cap {
                    // too many links: keep the best spread-out set
                    let base = v.get(n);
                    let mut c: Vec<D> = list.iter().map(|&x| D(1.0 - dot(base, v.get(x)), x)).collect();
                    c.sort();
                    *list = Self::select(v, &c, cap);
                }
            }
            eps = cands.iter().map(|d| d.1).collect();
        }
        if level > top {
            let mut e = self.entry.lock();
            if level > e.1 {
                *e = (node, level);
            }
        }
    }

    pub fn build(v: &Vectors, p: Params) -> Hnsw {
        let n = v.len();
        let ml = 1.0 / (p.m as f64).ln();
        // random level per node: each layer up has ~1/m as many nodes
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let levels: Vec<usize> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
                (-u.ln() * ml) as usize
            })
            .collect();
        let h = Hnsw {
            m: p.m,
            m0: p.m * 2,
            ef_build: p.ef_build,
            links: levels.iter().map(|&l| RwLock::new(vec![Vec::new(); l + 1])).collect(),
            entry: Mutex::new((0, levels[0])),
        };
        let done = AtomicUsize::new(0);
        (1..n as u32).into_par_iter().for_each_init(
            || Visited::new(n),
            |seen, i| {
                h.insert(v, i, levels[i as usize], seen);
                let d = done.fetch_add(1, Ordering::Relaxed);
                if d % 200_000 == 0 {
                    eprintln!("{d} / {n}");
                }
            },
        );
        h
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        let (ep, top) = *self.entry.lock();
        for x in [self.m as u32, self.ef_build as u32, self.links.len() as u32, ep, top as u32] {
            w.write_all(&x.to_le_bytes())?;
        }
        for l in &self.links {
            let l = l.read();
            w.write_all(&[l.len() as u8])?;
            for layer in l.iter() {
                w.write_all(&(layer.len() as u16).to_le_bytes())?;
                for &x in layer {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Hnsw> {
        let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
        let mut u32_ = || -> Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let (m, ef_build, n, ep, top) = (u32_()?, u32_()?, u32_()?, u32_()?, u32_()?);
        let mut links = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut b1 = [0u8; 1];
            r.read_exact(&mut b1)?;
            let mut node = Vec::with_capacity(b1[0] as usize);
            for _ in 0..b1[0] {
                let mut b2 = [0u8; 2];
                r.read_exact(&mut b2)?;
                let mut buf = vec![0u8; 4 * u16::from_le_bytes(b2) as usize];
                r.read_exact(&mut buf)?;
                node.push(buf.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect());
            }
            links.push(RwLock::new(node));
        }
        Ok(Hnsw { m: m as usize, m0: 2 * m as usize, ef_build: ef_build as usize, links, entry: Mutex::new((ep, top as usize)) })
    }

    /// The graph never changes once built, so search doesn't need locks or
    /// per-node allocations: layer 0, where nearly all the work happens,
    /// becomes one flat array with a fixed slot per node.
    pub fn freeze(self) -> Frozen {
        let stride = self.m0 + 1;
        let n = self.links.len();
        let mut l0 = vec![0u32; n * stride];
        let mut upper = Vec::with_capacity(n);
        for (i, l) in self.links.into_iter().enumerate() {
            let mut l = l.into_inner();
            let slot = &mut l0[i * stride..(i + 1) * stride];
            slot[0] = l[0].len() as u32;
            slot[1..1 + l[0].len()].copy_from_slice(&l[0]);
            l.remove(0);
            upper.push(l);
        }
        let (entry, top) = self.entry.into_inner();
        Frozen { stride, l0, upper, entry, top }
    }
}

pub struct Frozen {
    stride: usize,
    l0: Vec<u32>,              // per node: [count, links...] padded to stride
    upper: Vec<Vec<Vec<u32>>>, // per node: links on layers 1.. (empty for most nodes)
    entry: u32,
    top: usize,
}

impl Frozen {
    #[inline]
    fn nbrs(&self, node: u32, layer: usize) -> &[u32] {
        if layer == 0 {
            let b = node as usize * self.stride;
            &self.l0[b + 1..b + 1 + self.l0[b] as usize]
        } else {
            &self.upper[node as usize][layer - 1]
        }
    }

    fn search_layer(&self, v: &Vectors, q: &[f32], start: u32, ef: usize, seen: &mut Visited) -> Vec<D> {
        seen.reset();
        seen.insert(start);
        let d0 = D(Hnsw::dist(v, q, start), start);
        let mut cand = BinaryHeap::from([Reverse(d0)]);
        let mut best = BinaryHeap::from([d0]);
        while let Some(Reverse(c)) = cand.pop() {
            if best.len() >= ef && c.0 > best.peek().unwrap().0 {
                break;
            }
            for &n in self.nbrs(c.1, 0) {
                if !seen.insert(n) {
                    continue;
                }
                let d = Hnsw::dist(v, q, n);
                if best.len() < ef || d < best.peek().unwrap().0 {
                    cand.push(Reverse(D(d, n)));
                    best.push(D(d, n));
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        best.into_sorted_vec()
    }

    pub fn search(&self, v: &Vectors, q: &[f32], k: usize, ef: usize, seen: &mut Visited) -> Vec<(u32, f32)> {
        let (mut cur, mut cur_d) = (self.entry, Hnsw::dist(v, q, self.entry));
        for layer in (1..=self.top).rev() {
            loop {
                let mut moved = false;
                for &n in self.nbrs(cur, layer) {
                    let d = Hnsw::dist(v, q, n);
                    if d < cur_d {
                        (cur, cur_d, moved) = (n, d, true);
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        self.search_layer(v, q, cur, ef.max(k), seen).into_iter().take(k).map(|d| (d.1, 1.0 - d.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_true_neighbours_on_random_data() {
        let (n, dim) = (3000, 32);
        let mut s = 7u64;
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            data.push(((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        for v in data.chunks_exact_mut(dim) {
            let len = dot(v, v).sqrt();
            v.iter_mut().for_each(|x| *x /= len);
        }
        let v = Vectors { dim, data };
        let h = Hnsw::build(&v, Params { m: 12, ef_build: 100 }).freeze();
        let mut seen = Visited::new(n);
        let mut hit = 0;
        for qi in 0..100u32 {
            let q = v.get(qi * 7);
            let mut exact: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, dot(q, v.get(i)))).collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1));
            let truth: Vec<u32> = exact[..10].iter().map(|x| x.0).collect();
            hit += h.search(&v, q, 10, 64, &mut seen).iter().filter(|x| truth.contains(&x.0)).count();
        }
        assert!(hit as f32 / 1000.0 > 0.95, "recall@10 {}", hit as f32 / 1000.0);
    }
}
