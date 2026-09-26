import type { Gray } from './image';

// Square structuring element of side 2r+1, applied separably (rows, then columns): O(n·r), not O(n·r²).
// dilate: a pixel turns on if ANY neighbour is on. erode: stays on only if ALL neighbours are on.
function pass(m: Gray, r: number, dilating: boolean): Gray {
  const { w, h } = m, tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
  const line = (get: (i: number) => number, set: (i: number, v: number) => void, len: number) => {
    for (let i = 0; i < len; i++) {
      let v = dilating ? 0 : 1;
      for (let k = Math.max(0, i - r); k <= Math.min(len - 1, i + r); k++)
        if (dilating ? get(k) : !get(k)) { v = dilating ? 1 : 0; break; }
      set(i, v);
    }
  };
  for (let y = 0; y < h; y++) line((k) => m.data[y * w + k], (i, v) => (tmp[y * w + i] = v), w);
  for (let x = 0; x < w; x++) line((k) => tmp[k * w + x], (i, v) => (out[i * w + x] = v), h);
  return { w, h, data: out };
}

export const dilate = (m: Gray, r: number) => pass(m, r, true);
export const erode = (m: Gray, r: number) => pass(m, r, false);
export const open = (m: Gray, r: number) => dilate(erode(m, r), r); // removes specks and thin hairs
export const close = (m: Gray, r: number) => erode(dilate(m, r), r); // bridges small gaps
