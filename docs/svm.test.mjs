// Tests for the in-browser linear SVM (word + character TF-IDF).
// Run with: node --test docs/svm.test.mjs
// The expectations come from scikit-learn: fixtures.json holds its predictions and decision
// values for the 1,672 held-out messages, and fixtures-analyzer.json holds the character
// n-grams that TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5)) produces for awkward inputs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { charNgrams, prepareSvm, scoreSvm, predictSvm, explainSvm } from './svm.mjs';

const read = name => readFileSync(new URL(`./${name}`, import.meta.url));
const raw = JSON.parse(read('model-svm.json'));
const model = prepareSvm(raw);
const cases = JSON.parse(read('fixtures.json'));
const analyzer = JSON.parse(read('fixtures-analyzer.json'));

test('char_wb n-grams pad each word with spaces and stop at the whole padded word', () => {
  // ' hi ' has 4 characters, so sizes 2, 3 and 4 are produced and size 5 is skipped (scikit-learn breaks out)
  assert.deepEqual(charNgrams('Hi', 2, 5), [' h', 'hi', 'i ', ' hi', 'hi ', ' hi ']);
  assert.deepEqual(charNgrams('', 2, 5), []);
  assert.deepEqual(charNgrams('a  b', 2, 2), [' a', 'a ', ' b', 'b ']);
});

test('char_wb n-grams count characters, not UTF-16 units', () => {
  assert.deepEqual(charNgrams('\u{1F600}', 2, 3), [' \u{1F600}', '\u{1F600} ', ' \u{1F600} ']);
});

test('char_wb n-grams match scikit-learn on every analyzer fixture', () => {
  assert.ok(analyzer.length >= 40);
  for (const c of analyzer) {
    assert.deepEqual(charNgrams(c.text, 2, 5), c.ngrams, JSON.stringify(c.text));
  }
});

test('fixtures hold the 1,672 held-out messages with scikit-learn SVM predictions', () => {
  assert.equal(cases.length, 1672);
  assert.ok(cases.every(c => c.svm === 0 || c.svm === 1));
});

test('JS SVM predictions match scikit-learn on every held-out message', () => {
  const mismatches = cases.filter(c => predictSvm(model, c.text) !== c.svm);
  assert.equal(mismatches.length, 0, JSON.stringify(mismatches.slice(0, 5)));
});

test('JS SVM decision values are within 1e-4 of scikit-learn on every message', () => {
  let worst = 0;
  for (const c of cases) worst = Math.max(worst, Math.abs(scoreSvm(model, c.text) - c.svm_score));
  assert.ok(worst < 1e-4, `largest difference ${worst}`);
});

test('explanation adds up to the score: intercept plus the contribution of every chunk', () => {
  for (const c of cases.slice(0, 200)) {
    const e = explainSvm(model, c.text);
    const total = e.intercept + e.chunks.reduce((s, ch) => s + ch.contribution, 0);
    assert.ok(Math.abs(total - e.score) < 1e-9, `${c.text}: ${total} vs ${e.score}`);
    assert.equal(e.label, predictSvm(model, c.text));
  }
});

test('explanation chunks follow the whitespace-separated pieces of the message', () => {
  const e = explainSvm(model, 'Free  prize\tnow');
  assert.deepEqual(e.chunks.map(ch => ch.text), ['Free', 'prize', 'now']);
});

test('a message with no usable text scores the intercept', () => {
  assert.equal(scoreSvm(model, ''), model.intercept);
  assert.equal(scoreSvm(model, '   \n '), model.intercept);
});

test('the model file stays small enough for mobile: under 450 KB gzipped', () => {
  const gz = gzipSync(read('model-svm.json')).length;
  assert.ok(gz < 450 * 1024, `gzipped size ${gz} bytes`);
});
