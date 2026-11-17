use rust_stemmers::{Algorithm, Stemmer};

// Lucene's English stopword list, so BM25 numbers are comparable to the
// Elasticsearch baseline in the BEIR paper.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no",
    "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with",
];

pub struct Tokenizer {
    stemmer: Stemmer,
}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer { stemmer: Stemmer::create(Algorithm::English) }
    }

    /// Lowercase, split on anything that isn't a letter or digit, drop
    /// stopwords, stem. "Running shoes" -> ["run", "shoe"].
    pub fn tokens(&self, text: &str, out: &mut Vec<String>) {
        out.clear();
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            if word.is_empty() {
                continue;
            }
            let lower = word.to_lowercase();
            if STOPWORDS.contains(&lower.as_str()) {
                continue;
            }
            out.push(self.stemmer.stem(&lower).into_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_stems_and_drops_stopwords() {
        let t = Tokenizer::new();
        let mut out = Vec::new();
        t.tokens("The Running shoes, of 2024!", &mut out);
        assert_eq!(out, vec!["run", "shoe", "2024"]);
    }
}
