import { expect, test } from 'vitest';
import { toCSV } from '../src/csv';

test('csv quotes commas/quotes and keeps column order', () => {
  expect(toCSV([{ file: 'a,b.jpg', area: 1.5 }, { file: 'say "hi"', area: 2 }]))
    .toBe('file,area\n"a,b.jpg",1.5\n"say ""hi""",2\n');
});

test('missing values become empty cells', () => {
  expect(toCSV([{ a: 1, b: undefined }])).toBe('a,b\n1,\n');
});

test('empty -> empty string', () => expect(toCSV([])).toBe(''));
