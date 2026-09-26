import { type Gray, type RGB, toValue } from './image';
import { fftMag } from './fft';
import type { Calibration } from './larva';

export const MIN_CONFIDENCE = 5;
const BAND = 10;
const DETREND = 60; // half-width of the moving average; must exceed the longest tick period
// Longer periods lose to texture and lighting (their energy grows at low frequency). 40 px covers 0.01 mm
// ticks at any zoom and 0.05 mm ticks up to 800 px/mm; above that the fine ticks are resolved anyway.
const MAX_PERIOD = 40; // rows averaged either side of a candidate row
const STEP = 6; // candidate rows are this far apart (< BAND, so a bar can't slip between two)

// The stage micrometer is a 1 mm bar with ticks every 0.01 mm, longer ones every 0.05 mm and every 0.1 mm.
// Evenly spaced = periodic, and the FFT finds periods: tick spacing (px) × ticks per mm = pixels per mm.
// Which tick set wins depends on focus and zoom, so we count how many periods the bar spans to tell.
// 1. For each horizontal band of rows, build a 1-D brightness profile and measure how strongly it is
//    periodic (FFT peak / median). The bar's band wins even when it is faint and the background is
//    textured, because texture is not periodic.
// 2. A parabola through the winning peak and its neighbours gives sub-bin accuracy.
// 3. Cross-check that the periodic stretch is as long as 100 ticks should be.
export function calibrateFromScale(img: RGB, tickSets: number[] = [100, 20, 10]) {
  const g = toValue(img);
  let best = spectrum(g, BAND);
  for (let y = BAND + STEP; y < g.h - BAND; y += STEP) {
    const s = spectrum(g, y);
    if (s.strength > best.strength) best = s;
  }
  const { mag, k, strength, detr, row } = best;
  const n = (mag.length - 1) * 2, a = mag[k - 1], b = mag[k], c = mag[k + 1];
  const periodPx = n / (k + (0.5 * (a - c)) / (a - 2 * b + c || 1));
  // The bar is 1 mm, so (periodic stretch ÷ period) must be one of the tick counts. Larva segments,
  // fabric or noise can make a strong peak, but not one spanning 10, 20 or 100 periods.
  const extentPx = periodicExtent(detr, periodPx), count = extentPx / periodPx;
  const ticksPerMm = tickSets.reduce((b, t) => (Math.abs(Math.log(count / t)) < Math.abs(Math.log(count / b)) ? t : b));
  const ratio = count / ticksPerMm;
  const confidence = ratio > 0.85 && ratio < 1.15 ? strength : 0;
  return { pxPerMm: periodPx * ticksPerMm, confidence, method: 'fft' as const, row, periodPx, extentPx, ticksPerMm };
}

// Profile of rows row±BAND → subtract moving average (removes slow lighting changes) → Hann window
// (removes edge artefacts) → FFT. Strength = biggest peak with period 3–MAX_PERIOD px ÷ median of that range.
function spectrum(g: Gray, row: number) {
  const { w, h } = g, prof = new Float64Array(w), detr = new Float64Array(w), sig = new Float64Array(w);
  const y0 = Math.max(0, row - BAND), y1 = Math.min(h, row + BAND);
  for (let y = y0; y < y1; y++) for (let x = 0; x < w; x++) prof[x] += g.data[y * w + x] / (y1 - y0);
  let run = 0;
  for (let x = 0; x < DETREND && x < w; x++) run += prof[x];
  for (let x = 0; x < w; x++) {
    // running window [x-DETREND, x+DETREND], clipped to the image
    if (x + DETREND < w) run += prof[x + DETREND];
    if (x - DETREND - 1 >= 0) run -= prof[x - DETREND - 1];
    detr[x] = prof[x] - run / (Math.min(w - 1, x + DETREND) - Math.max(0, x - DETREND) + 1);
    sig[x] = detr[x] * (0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (w - 1)));
  }
  const mag = fftMag(sig), n = (mag.length - 1) * 2, lo = Math.ceil(n / MAX_PERIOD), hi = Math.floor(n / 3);
  let k = lo;
  for (let i = lo; i <= hi; i++) if (mag[i] > mag[k]) k = i;
  // Narrow ticks put almost as much energy at 2×, 3×… the tick frequency as at the tick frequency itself,
  // so the tallest peak can be a harmonic. Prefer the lowest sub-harmonic that is still at least half as strong.
  for (const d of [5, 4, 3, 2]) {
    const j = Math.round(k / d);
    if (j - 1 < lo) continue;
    const m = [j - 1, j, j + 1].reduce((b, i) => (mag[i] > mag[b] ? i : b));
    if (mag[m] >= 0.5 * mag[k]) { k = m; break; }
  }
  const sorted = Array.from(mag.slice(lo, hi + 1)).sort((p, q) => p - q);
  return { row, mag, k, detr, strength: mag[k] / (sorted[sorted.length >> 1] || 1) };
}

// Length of the longest run where the signal oscillates at `period`: multiply by a cosine and a sine
// at that period (demodulation), average over a few periods, and threshold the resulting amplitude.
function periodicExtent(sig: Float64Array, period: number): number {
  const w = sig.length, win = Math.max(4, Math.round(4 * period)), amp = new Float64Array(w);
  const cs = new Float64Array(w + 1), sn = new Float64Array(w + 1); // prefix sums
  for (let x = 0; x < w; x++) {
    const ph = (2 * Math.PI * x) / period;
    cs[x + 1] = cs[x] + sig[x] * Math.cos(ph);
    sn[x + 1] = sn[x] + sig[x] * Math.sin(ph);
  }
  let max = 0;
  for (let x = 0; x < w; x++) {
    const a = Math.max(0, x - (win >> 1)), z = Math.min(w, a + win);
    amp[x] = Math.hypot(cs[z] - cs[a], sn[z] - sn[a]) / (z - a);
    max = Math.max(max, amp[x]);
  }
  let best = 0, run = 0;
  for (let x = 0; x < w; x++) {
    run = amp[x] > 0.25 * max ? run + 1 : 0;
    best = Math.max(best, run);
  }
  // The averaging window blurs each end outwards by ~win/4 above the 25% line; take that back off.
  return Math.max(0, best - win / 2);
}

// Fallback: the user clicks both ends of the bar.
export function manualCalibration(a: [number, number], b: [number, number], mm = 1): Calibration {
  return { pxPerMm: Math.hypot(b[0] - a[0], b[1] - a[1]) / mm, confidence: Infinity, method: 'manual' };
}
