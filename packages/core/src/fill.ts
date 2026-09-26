import type { Gray } from './image';

// Flood the background from every border pixel. Background the flood can't reach is a hole → fill it.
export function fillHoles(m: Gray): Gray {
  const { w, h } = m, seen = new Uint8Array(w * h), stack: number[] = [];
  const push = (i: number) => {
    if (!m.data[i] && !seen[i]) { seen[i] = 1; stack.push(i); }
  };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const i = stack.pop()!, x = i % w, y = (i - x) / w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  return { w, h, data: m.data.map((v, i) => (v || !seen[i] ? 1 : 0)) };
}
