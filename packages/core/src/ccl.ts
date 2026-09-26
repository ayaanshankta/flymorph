import type { Gray } from './image';

export type Component = { id: number; area: number; x: number; y: number; w: number; h: number; cx: number; cy: number; edge: boolean };

// Connected-component labelling by flood fill (8-connected: diagonals count as touching).
// labels[i] = component id (1, 2, …) or 0 for background. Ids follow raster order of each blob's first pixel.
export function components(m: Gray): { labels: Int32Array; comps: Component[] } {
  const { w, h } = m, labels = new Int32Array(w * h), comps: Component[] = [], stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (!m.data[s] || labels[s]) continue;
    const id = comps.length + 1;
    let area = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = 0, y1 = 0;
    labels[s] = id;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!, x = i % w, y = (i - x) / w;
      area++; sx += x; sy += y;
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy, j = ny * w + nx;
          if (nx >= 0 && ny >= 0 && nx < w && ny < h && m.data[j] && !labels[j]) { labels[j] = id; stack.push(j); }
        }
    }
    comps.push({ id, area, x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, cx: sx / area, cy: sy / area,
      edge: x0 === 0 || y0 === 0 || x1 === w - 1 || y1 === h - 1 });
  }
  return { labels, comps };
}
