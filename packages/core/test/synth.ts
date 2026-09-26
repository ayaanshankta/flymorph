import type { RGB, Gray } from '../src/image';

export function rgb(w: number, h: number, f: (x: number, y: number) => [number, number, number]): RGB {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [r, g, b] = f(x, y), i = (y * w + x) * 4;
    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
  }
  return { w, h, data };
}

export function mask(w: number, h: number, f: (x: number, y: number) => boolean): Gray {
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = f(x, y) ? 1 : 0;
  return { w, h, data };
}

export const ellipse = (cx: number, cy: number, a: number, b: number) =>
  (x: number, y: number) => ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1;
