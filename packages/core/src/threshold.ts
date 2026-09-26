import type { Gray } from './image';

// Otsu: try every threshold, keep the one that best separates the histogram into two groups
// (maximises between-class variance). Adapts to exposure instead of a hardcoded cut-off.
export function otsu(g: Gray): number {
  const hist = new Float64Array(256);
  for (const v of g.data) hist[v]++;
  const n = g.data.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let wB = 0, sumB = 0, best = 0, t = 0;
  for (let i = 0; i < 256; i++) {
    wB += hist[i];
    if (!wB) continue;
    const wF = n - wB;
    if (!wF) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sum - sumB) / wF, between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; t = i; }
  }
  return t;
}

export function binarize(g: Gray, t: number): Gray {
  return { w: g.w, h: g.h, data: g.data.map((v) => (v > t ? 1 : 0)) };
}
