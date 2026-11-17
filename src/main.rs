mod bm25;
mod index;
mod tokenize;
mod varint;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};

const USAGE: &str = "usage:
  wiki-search build <corpus.jsonl> <index dir>
  wiki-search run <index dir> <queries.jsonl> <run.tsv> [k]   score every query, write a TREC run
  wiki-search query <index dir> <words...>";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") if args.len() == 3 => {
            let t = Instant::now();
            index::build(Path::new(&args[1]), Path::new(&args[2]))?;
            eprintln!("built in {:.0?}", t.elapsed());
        }
        Some("run") if args.len() >= 4 => {
            let k = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(100);
            run(Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]), k)?;
        }
        Some("query") if args.len() >= 3 => {
            let idx = index::Index::open(Path::new(&args[1]))?;
            let mut s = bm25::Searcher::new(&idx);
            let t = Instant::now();
            let hits = s.search(&args[2..].join(" "), 10);
            eprintln!("{:.1?}", t.elapsed());
            for (doc, score) in hits {
                println!("{score:.3}\t{}", idx.ids[doc as usize]);
            }
        }
        _ => bail!("{USAGE}"),
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct Query {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// Runs every query and writes "qid docid rank score" lines, plus latency.
fn run(dir: &Path, queries: &Path, out: &Path, k: usize) -> Result<()> {
    let idx = index::Index::open(dir)?;
    let mut s = bm25::Searcher::new(&idx);
    let mut w = BufWriter::new(File::create(out)?);
    let mut times = Vec::new();
    for line in BufReader::new(File::open(queries)?).lines() {
        let q: Query = serde_json::from_str(&line?)?;
        let t = Instant::now();
        let hits = s.search(&q.text, k);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        for (rank, (doc, score)) in hits.iter().enumerate() {
            writeln!(w, "{}\t{}\t{}\t{:.4}", q.id, idx.ids[*doc as usize], rank + 1, score)?;
        }
    }
    w.flush()?;
    times.sort_by(f64::total_cmp);
    let pct = |p: f64| times[((times.len() as f64 * p) as usize).min(times.len() - 1)];
    eprintln!("{} queries  p50 {:.2} ms  p99 {:.2} ms", times.len(), pct(0.5), pct(0.99));
    Ok(())
}
