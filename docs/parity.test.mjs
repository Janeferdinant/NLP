// Parity test: the JavaScript classifier must agree with scikit-learn on every
// held-out test message. Run with: node --test docs/parity.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { predict } from './predict.mjs';

const model = JSON.parse(readFileSync(new URL('./model.json', import.meta.url)));
const cases = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url)));

test('fixtures hold the 1,672 held-out test messages', () => {
  assert.equal(cases.length, 1672);
});

test('JS predictions match sklearn on every held-out message', () => {
  const mismatches = cases.filter(c => predict(model, c.text) !== c.sklearn);
  assert.equal(mismatches.length, 0, JSON.stringify(mismatches.slice(0, 5)));
});
