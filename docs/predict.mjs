// In-browser re-implementation of the notebook's scikit-learn pipeline:
// TfidfVectorizer(lowercase=True, stop_words='english') followed by MultinomialNB().
// The model parameters come from docs/model.json (written by scripts/export_model.py).
// docs/parity.test.mjs checks that predict() agrees with scikit-learn on all
// 1,672 held-out test messages.

// scikit-learn's default token_pattern is (?u)\b\w\w+\b. Python's \w matches
// Unicode letters, Unicode numbers and "_", so this is the JavaScript equivalent.
const DEFAULT_TOKEN_PATTERN = '(?u)\\b\\w\\w+\\b';
const TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;

function checkParams(params) {
  if (params.token_pattern !== DEFAULT_TOKEN_PATTERN) {
    throw new Error(`Unsupported token_pattern: ${params.token_pattern}`);
  }
}

// Turn a message into vocabulary terms, in the same order as sklearn's analyzer.
export function analyze(model, text) {
  const { lowercase, ngram_range: [minN, maxN] } = model.params;
  checkParams(model.params);
  const doc = lowercase ? text.toLowerCase() : text;
  const stop = model._stopSet ?? (model._stopSet = new Set(model.stop_words));
  const tokens = (doc.match(TOKEN_RE) ?? []).filter(t => !stop.has(t));
  if (maxN === 1) return tokens;
  const terms = minN === 1 ? [...tokens] : [];
  for (let n = Math.max(minN, 2); n <= Math.min(maxN, tokens.length); n++) {
    for (let i = 0; i + n <= tokens.length; i++) terms.push(tokens.slice(i, i + n).join(' '));
  }
  return terms;
}

// TF-IDF vector as a list of [featureIndex, weight] pairs, sorted by feature index
// (the order scipy stores them in, so sums are accumulated in the same order).
export function vectorize(model, text) {
  const counts = new Map();
  for (const term of analyze(model, text)) {
    const j = model.vocabulary[term];
    if (j !== undefined) counts.set(j, (counts.get(j) ?? 0) + 1);
  }
  const { sublinear_tf, norm } = model.params;
  const entries = [...counts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([j, c]) => [j, (sublinear_tf ? Math.log(c) + 1 : c) * model.idf[j]]);
  if (norm === 'l2' || norm === 'l1') {
    let total = 0;
    for (const [, w] of entries) total += norm === 'l2' ? w * w : Math.abs(w);
    const scale = norm === 'l2' ? Math.sqrt(total) : total;
    if (scale > 0) for (const e of entries) e[1] /= scale;
  }
  return entries;
}

// Joint log-likelihood per class: x . feature_log_prob[c] + class_log_prior[c]
export function jointLogLikelihood(model, x) {
  return model.classes.map((_, c) => {
    const flp = model.feature_log_prob[c];
    let dot = 0;
    for (const [j, w] of x) dot += w * flp[j];
    return dot + model.class_log_prior[c];
  });
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// Returns the predicted class label (for this model: 0 = ham, 1 = spam).
export function predict(model, text) {
  return model.classes[argmax(jointLogLikelihood(model, vectorize(model, text)))];
}

// Prediction plus the words behind it, for the "How it works" panel.
// For two classes, each word's contribution to the log-odds of class 1 over class 0
// is weight * (feature_log_prob[1][j] - feature_log_prob[0][j]).
export function explain(model, text) {
  const x = vectorize(model, text);
  const jll = jointLogLikelihood(model, x);
  const best = argmax(jll);
  if (!model._terms) {
    model._terms = [];
    for (const [t, j] of Object.entries(model.vocabulary)) model._terms[j] = t;
  }
  const words = x.map(([j, w]) => ({
    word: model._terms[j],
    weight: w,
    contribution: w * (model.feature_log_prob[1][j] - model.feature_log_prob[0][j]),
  }));
  const doc = model.params.lowercase ? text.toLowerCase() : text;
  const tokens = doc.match(TOKEN_RE) ?? [];
  const stop = model._stopSet ?? (model._stopSet = new Set(model.stop_words));
  return {
    label: model.classes[best],
    jointLogLikelihood: jll,
    priorLogOdds: model.class_log_prior[1] - model.class_log_prior[0],
    logOdds: jll[1] - jll[0],
    words,
    tokenCount: tokens.length,
    stopWordCount: tokens.filter(t => stop.has(t)).length,
    unknownCount: tokens.filter(t => !stop.has(t) && model.vocabulary[t] === undefined).length,
  };
}
