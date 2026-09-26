import { expect, test } from 'vitest';
import { anatomyIssues, misorderedFlies } from '../src/anatomy';

// Draw a vertical fly in a 40×60 grid, column 10–29. parts = [label, firstRow, lastRow) top to bottom.
function fly(parts: [number, number, number][], w = 40, h = 60) {
  const l = new Uint8Array(w * h);
  for (const [k, y0, y1] of parts) for (let y = y0; y < y1; y++) for (let x = 10; x < 30; x++) l[y * w + x] = k;
  return { l, w, h };
}
const issues = (f: ReturnType<typeof fly>) => anatomyIssues(f.l, f.w, f.h).flatMap((i) => i.issues);

test('a normal fly (head, thorax, abdomen touching, head smallest) has no issues', () => {
  expect(issues(fly([[1, 2, 10], [2, 10, 25], [3, 25, 50]]))).toEqual([]);
});

test('thorax not between head and abdomen → order', () => {
  expect(issues(fly([[1, 2, 10], [3, 10, 30], [2, 30, 50]]))).toContain('order');
});

test('a big gap between thorax and abdomen → gap', () => {
  expect(issues(fly([[1, 2, 10], [2, 10, 22], [3, 30, 58]]))).toContain('gap'); // 8 empty rows
});

test('a small crack is not a gap', () => {
  expect(issues(fly([[1, 2, 10], [2, 10, 24], [3, 25, 50]]))).not.toContain('gap');
});

test('head bigger than the thorax → sizes', () => {
  expect(issues(fly([[1, 2, 22], [2, 22, 30], [3, 30, 55]]))).toContain('sizes');
});

test('a part split into two big pieces → pieces', () => {
  const f = fly([[1, 2, 10], [2, 10, 25], [3, 25, 50]]);
  for (let y = 25; y < 50; y++) f.l[y * f.w + 19] = f.l[y * f.w + 20] = 2; // thorax strip splits the abdomen in two
  expect(issues(f)).toContain('pieces');
});

test('a fly missing a part is not judged', () => {
  expect(anatomyIssues(new Uint8Array(100).fill(2), 10, 10)).toEqual([]);
});

test('misorderedFlies still reports only order problems', () => {
  const f = fly([[1, 2, 10], [3, 10, 30], [2, 30, 50]]);
  expect(misorderedFlies(f.l, f.w, f.h)).toHaveLength(1);
});
