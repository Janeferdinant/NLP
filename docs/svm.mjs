// In-browser re-implementation of the best model in the repository: scikit-learn's
//   FeatureUnion([TfidfVectorizer(lowercase=True, stop_words='english'),
//                 TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, min_df=2)])
// followed by LinearSVC(). The parameters come from docs/model-svm.json (written by
// scripts/export_model.py). docs/svm.test.mjs checks the n-grams against scikit-learn and the
// predictions on all 1,672 held-out test messages.
//
// Details that must match scikit-learn exactly:
//  * Python's str.split() and re \s treat U+001C..U+001F and U+0085 as white space and not U+FEFF;
//    JavaScript's \s does the opposite, so the white-space class is spelled out below.
//  * char_wb n-grams are cut from each white-space-separated word padded with one space on each side,
//    by characters (code points), not UTF-16 units. A word shorter than n yields one n-gram of the whole
//    padded word for that n.
//  * Each of the two vectorizers has its own vocabulary, idf, sublinear-tf setting and L2 norm; the
//    coefficient vector lists the word features first, then the character features.
import { analyze } from './predict.mjs';

const WS = '\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const SPLIT_RE = new RegExp(`[${WS}]+`, 'u');
const CHUNK_RE = new RegExp(`[^${WS}]+`, 'gu');

// Character n-grams inside word boundaries (scikit-learn's analyzer='char_wb').
export function charNgrams(text, minN, maxN, lowercase = true) {
  const doc = lowercase ? text.toLowerCase() : text;
  const grams = [];
  for (const word of doc.split(SPLIT_RE)) {
    if (!word) continue;
    const chars = Array.from(` ${word} `);
    const len = chars.length;
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      grams.push(chars.slice(offset, offset + n).join(''));
      while (offset + n < len) {
        offset += 1;
        grams.push(chars.slice(offset, offset + n).join(''));
      }
      if (offset === 0) break; // a short word counts once
    }
  }
  return grams;
}

function prepareVocab(part) {
  const index = new Map(part.terms.map((t, j) => [t, j]));
  // scikit-learn's smoothed idf, rebuilt from the integer document frequencies
  const idf = Float64Array.from(part.df, df => Math.log((1 + part.n_docs) / (1 + df)) + 1);
  return { ...part, index, idf };
}

// Turn the parsed model-svm.json into a model ready for scoring.
export function prepareSvm(raw) {
  return {
    intercept: raw.intercept,
    classes: raw.classes,
    coef: Float64Array.from(raw.coef),
    word: prepareVocab(raw.word),
    char: prepareVocab({ ...raw.char, params: { ...raw.char.params } }),
    charOffset: raw.word.terms.length,
  };
}

function wordTerms(part, text) {
  return analyze(part, text);
}

function charTerms(part, text) {
  const { ngram_range: [minN, maxN], lowercase } = part.params;
  return charNgrams(text, minN, maxN, lowercase);
}

// TF-IDF vector of one vectorizer: Map featureIndex -> { weight, count }.
function tfidf(part, terms) {
  const counts = new Map();
  for (const term of terms) {
    const j = part.index.get(term);
    if (j !== undefined) counts.set(j, (counts.get(j) ?? 0) + 1);
  }
  const vec = new Map();
  let total = 0;
  for (const [j, c] of counts) {
    const w = (part.params.sublinear_tf ? Math.log(c) + 1 : c) * part.idf[j];
    vec.set(j, { weight: w, count: c });
    total += w * w;
  }
  const scale = part.params.norm === 'l2' ? Math.sqrt(total) : 1;
  if (scale > 0) for (const v of vec.values()) v.weight /= scale;
  return vec;
}

function vectors(model, text) {
  return {
    word: tfidf(model.word, wordTerms(model.word, text)),
    char: tfidf(model.char, charTerms(model.char, text)),
  };
}

// The SVM decision value: above 0 means spam.
export function scoreSvm(model, text) {
  const v = vectors(model, text);
  let dot = 0;
  for (const [j, { weight }] of v.word) dot += model.coef[j] * weight;
  for (const [j, { weight }] of v.char) dot += model.coef[model.charOffset + j] * weight;
  return dot + model.intercept;
}

// Class label: 1 = spam, 0 = ham.
export function predictSvm(model, text) {
  return scoreSvm(model, text) > 0 ? model.classes[1] : model.classes[0];
}

// The decision value split over the white-space-separated pieces of the message ("chunks").
// Every feature belongs to exactly one chunk (a word's characters, or the word itself), so
//   score = intercept + sum of chunk contributions.
// A feature that occurs c times in the message splits its contribution equally over its occurrences.
export function explainSvm(model, text) {
  const v = vectors(model, text);
  const share = (part, vec, offset, chunkText) => {
    const terms = part === model.word ? wordTerms(part, chunkText) : charTerms(part, chunkText);
    let sum = 0;
    let known = 0;
    for (const term of terms) {
      const j = part.index.get(term);
      if (j === undefined) continue;
      const { weight, count } = vec.get(j);
      sum += (model.coef[offset + j] * weight) / count;
      known += 1;
    }
    return { sum, known };
  };
  const chunks = [];
  for (const m of text.matchAll(CHUNK_RE)) {
    const w = share(model.word, v.word, 0, m[0]);
    const c = share(model.char, v.char, model.charOffset, m[0]);
    chunks.push({
      text: m[0],
      index: m.index,
      contribution: w.sum + c.sum,
      wordContribution: w.sum,
      charContribution: c.sum,
      knownFeatures: w.known + c.known,
    });
  }
  const score = scoreSvm(model, text);
  return {
    label: score > 0 ? model.classes[1] : model.classes[0],
    score,
    intercept: model.intercept,
    chunks,
    wordFeatures: v.word.size,
    charFeatures: v.char.size,
  };
}
