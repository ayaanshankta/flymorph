import { type RGB, toValue, gaussianBlur } from './image';
import { otsu, binarize } from './threshold';
import { open, close } from './morph';
import { fillHoles } from './fill';
import { components, type Component } from './ccl';

export type Calibration = { pxPerMm: number; confidence: number; method: 'fft' | 'manual' };
export type Measurement = {
  id: number;
  part: 'larva' | 'head' | 'thorax' | 'abdomen' | 'body';
  areaPx: number;
  areaMm2: number;
  sdMm2?: number;
  bbox: [x: number, y: number, w: number, h: number];
  flag?: string;
};
export type LarvaOptions = { minAreaMm2: number; minAspect: number; minFill: number; minThreshold: number };
const DEFAULTS: LarvaOptions = { minAreaMm2: 0.05, minAspect: 1.3, minFill: 0.25, minThreshold: 60 };
// A single larva is nearly convex: real ones measured 0.84–0.98. Blobs below 0.8 were larvae fused with
// background objects (e.g. a tape strip) or with each other — worth a human look.
const MIN_SOLIDITY = 0.8;

// Solidity = blob area ÷ area of its convex hull (the shape a rubber band around it would make).
// Hull of the pixel corners at each row's leftmost and rightmost pixel (monotone chain), area by shoelace.
export function solidity(labels: Int32Array, width: number, c: Component): number {
  const pts: [number, number][] = [];
  for (let y = c.y; y < c.y + c.h; y++) {
    let l = -1, r = -1;
    for (let x = c.x; x < c.x + c.w; x++) if (labels[y * width + x] === c.id) { if (l < 0) l = x; r = x; }
    if (l >= 0) pts.push([l, y], [l, y + 1], [r + 1, y], [r + 1, y + 1]);
  }
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (ps: [number, number][]) => {
    const h: [number, number][] = [];
    for (const p of ps) { while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop(); h.push(p); }
    return h.slice(0, -1);
  };
  const hull = [...half(pts), ...half([...pts].reverse())];
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i], [x2, y2] = hull[(i + 1) % hull.length];
    area += x1 * y2 - x2 * y1;
  }
  return c.area / (Math.abs(area) / 2);
}

export function measureLarvae(img: RGB, cal: Calibration, opt: Partial<LarvaOptions> = {}) {
  const o = { ...DEFAULTS, ...opt };
  const v = gaussianBlur(toValue(img));
  // Larvae are bright on dark. Otsu adapts to exposure; minThreshold stops it chasing background noise
  // when an image has no larvae at all.
  const t = Math.max(otsu(v), o.minThreshold);
  const r = Math.max(1, Math.round(img.w / 1400)); // ≈ the old 5 px / 9 px kernels at 3584 px wide
  const mask = fillHoles(close(open(binarize(v, t), r), 2 * r));
  const { labels, comps } = components(mask);
  const px2 = cal.pxPerMm ** 2;

  const kept = comps.filter((c) => {
    const aspect = Math.max(c.w, c.h) / Math.max(1, Math.min(c.w, c.h)); // orientation-free (old code used h/w)
    return c.area / px2 >= o.minAreaMm2 && aspect >= o.minAspect && c.area / (c.w * c.h) >= o.minFill;
  });
  const keep = new Set(kept.map((c) => c.id));
  for (let i = 0; i < labels.length; i++) if (!keep.has(labels[i])) labels[i] = 0;

  const measurements: Measurement[] = kept.map((c, k) => {
    const flags = [c.edge && 'edge', solidity(labels, img.w, c) < MIN_SOLIDITY && 'irregular'].filter(Boolean);
    return {
      id: k + 1, part: 'larva', areaPx: c.area, areaMm2: c.area / px2, bbox: [c.x, c.y, c.w, c.h],
      ...(flags.length ? { flag: flags.join(',') } : {}),
    };
  });
  return { measurements, labels };
}
