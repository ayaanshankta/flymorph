import { type RGB, toValue, gaussianBlur } from './image';
import { otsu, binarize } from './threshold';
import { open, close } from './morph';
import { fillHoles } from './fill';
import { components } from './ccl';

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
  const sorted = kept.map((c) => c.area).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const keep = new Set(kept.map((c) => c.id));
  for (let i = 0; i < labels.length; i++) if (!keep.has(labels[i])) labels[i] = 0;

  const measurements: Measurement[] = kept.map((c, k) => {
    const flags = [c.edge && 'edge', kept.length >= 3 && c.area > 1.8 * median && 'merged?'].filter(Boolean);
    return {
      id: k + 1, part: 'larva', areaPx: c.area, areaMm2: c.area / px2, bbox: [c.x, c.y, c.w, c.h],
      ...(flags.length ? { flag: flags.join(',') } : {}),
    };
  });
  return { measurements, labels };
}
