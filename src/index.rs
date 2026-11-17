//! Inverted index: for every term, the list of documents containing it and
//! how many times. On disk:
//!
//!   terms.bin     sorted terms: [len u8][bytes][df u32][offset u64][bytes u32]
//!   postings.bin  per term: varint(doc gap), varint(tf), ...
//!   docs.bin      [n u32][length of each doc in tokens, u32 each]
//!   ids.txt       the corpus id of each doc, one per line

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::Deserialize;

use crate::tokenize::Tokenizer;
use crate::varint;

#[derive(Deserialize)]
struct Doc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Clone, Copy)]
pub struct TermInfo {
    pub df: u32,
    offset: u64,
    len: u32,
}

pub struct Index {
    pub terms: HashMap<String, TermInfo>,
    postings: Mmap,
    pub doc_len: Vec<u32>,
    pub avg_len: f32,
    pub ids: Vec<String>,
}

/// One pass over the corpus, building every postings list in memory, then
/// written out sorted by term. 2.7M Wikipedia passages fit in a few GB.
pub fn build(corpus: &Path, out: &Path) -> Result<()> {
    fs::create_dir_all(out)?;
    let tok = Tokenizer::new();
    let mut lists: HashMap<String, Vec<u32>> = HashMap::new(); // doc, tf, doc, tf...
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut doc_len = Vec::new();
    let mut ids = BufWriter::new(File::create(out.join("ids.txt"))?);
    let mut toks = Vec::new();

    let reader = BufReader::with_capacity(1 << 20, File::open(corpus).context("corpus")?);
    for (n, line) in reader.lines().enumerate() {
        let doc: Doc = serde_json::from_str(&line?)?;
        let doc_id = n as u32;
        writeln!(ids, "{}", doc.id)?;
        tok.tokens(&format!("{} {}", doc.title, doc.text), &mut toks);
        doc_len.push(toks.len() as u32);
        counts.clear();
        for t in toks.drain(..) {
            *counts.entry(t).or_insert(0) += 1;
        }
        for (t, tf) in counts.drain() {
            let list = lists.entry(t).or_default();
            list.push(doc_id);
            list.push(tf);
        }
        if n % 500_000 == 0 {
            eprintln!("{n} docs, {} terms", lists.len());
        }
    }
    ids.flush()?;

    let mut terms: Vec<_> = lists.into_iter().collect();
    terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut tw = BufWriter::new(File::create(out.join("terms.bin"))?);
    let mut pw = BufWriter::new(File::create(out.join("postings.bin"))?);
    let mut offset = 0u64;
    let mut buf = Vec::new();
    for (term, list) in &terms {
        buf.clear();
        let mut prev = 0;
        for pair in list.chunks_exact(2) {
            varint::write(&mut buf, pair[0] - prev);
            varint::write(&mut buf, pair[1]);
            prev = pair[0];
        }
        let tb = term.as_bytes();
        let tb = &tb[..tb.len().min(255)];
        tw.write_all(&[tb.len() as u8])?;
        tw.write_all(tb)?;
        tw.write_all(&((list.len() / 2) as u32).to_le_bytes())?;
        tw.write_all(&offset.to_le_bytes())?;
        tw.write_all(&(buf.len() as u32).to_le_bytes())?;
        pw.write_all(&buf)?;
        offset += buf.len() as u64;
    }
    tw.flush()?;
    pw.flush()?;

    let mut dw = BufWriter::new(File::create(out.join("docs.bin"))?);
    dw.write_all(&(doc_len.len() as u32).to_le_bytes())?;
    for l in &doc_len {
        dw.write_all(&l.to_le_bytes())?;
    }
    dw.flush()?;
    eprintln!("{} docs, {} terms, {} MB of postings", doc_len.len(), terms.len(), offset >> 20);
    Ok(())
}

impl Index {
    pub fn open(dir: &Path) -> Result<Index> {
        let mut raw = Vec::new();
        File::open(dir.join("terms.bin"))?.read_to_end(&mut raw)?;
        let mut terms = HashMap::new();
        let mut p = 0;
        while p < raw.len() {
            let l = raw[p] as usize;
            let term = String::from_utf8_lossy(&raw[p + 1..p + 1 + l]).into_owned();
            p += 1 + l;
            let df = u32::from_le_bytes(raw[p..p + 4].try_into()?);
            let offset = u64::from_le_bytes(raw[p + 4..p + 12].try_into()?);
            let len = u32::from_le_bytes(raw[p + 12..p + 16].try_into()?);
            p += 16;
            terms.insert(term, TermInfo { df, offset, len });
        }

        let mut raw = Vec::new();
        File::open(dir.join("docs.bin"))?.read_to_end(&mut raw)?;
        let n = u32::from_le_bytes(raw[0..4].try_into()?) as usize;
        let doc_len: Vec<u32> = raw[4..4 + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let avg_len = doc_len.iter().map(|&l| l as f64).sum::<f64>() as f32 / n as f32;

        let ids = fs::read_to_string(dir.join("ids.txt"))?.lines().map(String::from).collect();
        // the OS pages postings in as they're read; nothing is copied up front
        let postings = unsafe { Mmap::map(&File::open(dir.join("postings.bin"))?)? };
        Ok(Index { terms, postings, doc_len, avg_len, ids })
    }

    pub fn num_docs(&self) -> usize {
        self.doc_len.len()
    }

    pub fn postings_of(&self, info: TermInfo) -> &[u8] {
        &self.postings[info.offset as usize..(info.offset + info.len as u64) as usize]
    }

    /// Walk a term's postings, calling f(doc, tf) for each.
    #[inline]
    pub fn for_each_posting(&self, info: TermInfo, mut f: impl FnMut(u32, u32)) {
        let bytes = self.postings_of(info);
        let mut pos = 0;
        let mut doc = 0;
        while pos < bytes.len() {
            doc += varint::read(bytes, &mut pos);
            let tf = varint::read(bytes, &mut pos);
            f(doc, tf);
        }
    }
}
