import { expect, test } from 'vitest';
import { calibrateFromScale, manualCalibration, MIN_CONFIDENCE } from '../src/calibrate';
import { fftMag } from '../src/fft';
import { rgb } from './synth';

test('fft of a pure cosine peaks at its frequency bin', () => {
  const s = Float64Array.from({ length: 64 }, (_, i) => Math.cos((2 * Math.PI * 5 * i) / 64));
  const m = fftMag(s);
  expect(m.indexOf(Math.max(...m))).toBe(5);
});

test('recovers a synthetic ruler period within 1%', () => {
  const period = 9.14; // → 914 px/mm
  const img = rgb(1200, 300, (x, y) => {
    const onBar = y > 140 && y < 160 && x > 200 && x < 1114;
    return onBar && (x - 200) % period < 2 ? [60, 60, 70] : [200, 210, 205];
  });
  const cal = calibrateFromScale(img);
  expect(Math.abs(cal.pxPerMm - 914) / 914).toBeLessThan(0.01);
  expect(cal.confidence).toBeGreaterThan(MIN_CONFIDENCE);
});

test('a noisy image with no ruler is low confidence', () => {
  let s = 1;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 255;
  expect(calibrateFromScale(rgb(512, 256, () => [rand(), rand(), rand()])).confidence).toBeLessThan(MIN_CONFIDENCE);
});

test('manual: two clicks 914 px apart on a 1 mm bar', () => {
  expect(manualCalibration([100, 50], [1014, 50]).pxPerMm).toBeCloseTo(914);
});

test('periodic stripes that are not a 100-tick bar are rejected', () => {
  // stripes across the whole width, period 39 px → 100 ticks would need 3900 px; image is 1200
  const img = rgb(1200, 300, (x, y) => (y > 140 && y < 160 && x % 39 < 6 ? [60, 60, 70] : [200, 210, 205]));
  expect(calibrateFromScale(img).confidence).toBeLessThan(MIN_CONFIDENCE);
});

test('a bar with only 30 ticks is rejected', () => {
  const img = rgb(1200, 300, (x, y) => (y > 140 && y < 160 && x > 200 && x < 474 && (x - 200) % 9.14 < 2 ? [60, 60, 70] : [200, 210, 205]));
  expect(calibrateFromScale(img).confidence).toBeLessThan(MIN_CONFIDENCE);
});

test('finds a faint ruler on a strongly textured background', () => {
  let s = 7;
  const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const img = rgb(1400, 600, (x, y) => {
    const onBar = y > 290 && y < 310 && x > 300 && x < 1214 && (x - 300) % 9.14 < 2;
    const v = onBar ? 185 : 200 - (y > 450 ? rand() * 120 : rand() * 10); // rough texture in the lower rows
    return [v, v, v];
  });
  const cal = calibrateFromScale(img);
  expect(Math.abs(cal.pxPerMm - 914) / 914).toBeLessThan(0.01);
  expect(cal.confidence).toBeGreaterThan(MIN_CONFIDENCE);
});

test('a ruler where only the 0.05 mm ticks are resolved still gives the right scale', () => {
  // 600 px/mm → 0.05 mm ticks every 30 px (what the real low-zoom scale images show)
  const img = rgb(1400, 300, (x, y) => (y > 140 && y < 160 && x > 200 && x < 800 && (x - 200) % 30 < 4 ? [60, 60, 70] : [200, 210, 205]));
  const cal = calibrateFromScale(img);
  expect(Math.abs(cal.pxPerMm - 600) / 600).toBeLessThan(0.01);
  expect(cal.confidence).toBeGreaterThan(MIN_CONFIDENCE);
});

test('~6 periodic stripes (like larva segments) are rejected', () => {
  const img = rgb(1400, 300, (x, y) => (y > 100 && y < 200 && x > 500 && x < 740 && (x - 500) % 38 < 12 ? [60, 60, 70] : [200, 210, 205]));
  expect(calibrateFromScale(img).confidence).toBeLessThan(MIN_CONFIDENCE);
});
