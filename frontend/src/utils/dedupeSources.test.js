import test from 'node:test';
import assert from 'node:assert/strict';
import { dedupeSources } from './dedupeSources.js';

const src = (chunk_id, title, excerpt, relevance_score = 0.8) => ({
  chunk_id,
  document_id: `doc-${chunk_id}`,
  title,
  excerpt,
  relevance_score,
});

test('removes sources repeated with the same chunk_id', () => {
  const a = src('c1', 'Employee Leave Policy', 'Sick leave: 10 days per year.');
  assert.deepEqual(dedupeSources([a, { ...a }, { ...a }]), [a]);
});

test('removes same title + excerpt stored under different chunk ids', () => {
  const list = [
    src('c1', 'Employee Leave Policy', 'Sick leave: 10 days per year.'),
    src('c2', 'Employee Leave Policy', 'Sick leave: 10 days per year.'),
    src('c3', 'Employee Leave Policy', '  sick LEAVE:  10 days per year. '),
  ];
  const out = dedupeSources(list);
  assert.equal(out.length, 1);
  assert.equal(out[0].chunk_id, 'c1');
});

test('keeps distinct chunks of the same document', () => {
  const list = [
    src('c1', 'Employee Leave Policy', 'Sick leave: 10 days per year.'),
    src('c2', 'Employee Leave Policy', 'Casual leave: 12 days per year.'),
  ];
  assert.equal(dedupeSources(list).length, 2);
});

test('keeps same excerpt under different document titles', () => {
  const list = [
    src('c1', 'Employee Leave Policy', 'Up to 10 days.'),
    src('c2', 'Contractor Policy', 'Up to 10 days.'),
  ];
  assert.equal(dedupeSources(list).length, 2);
});

test('preserves order and relevance scores of the kept sources', () => {
  const list = [
    src('c1', 'A', 'one', 0.9),
    src('c2', 'B', 'two', 0.7),
    src('c1', 'A', 'one', 0.9),
  ];
  assert.deepEqual(
    dedupeSources(list).map((s) => [s.chunk_id, s.relevance_score]),
    [['c1', 0.9], ['c2', 0.7]],
  );
});

test('handles empty / invalid input', () => {
  assert.deepEqual(dedupeSources([]), []);
  assert.deepEqual(dedupeSources(undefined), []);
  assert.deepEqual(dedupeSources(null), []);
});
