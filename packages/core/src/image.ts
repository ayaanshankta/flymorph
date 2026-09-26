// Images are plain typed arrays so the same code runs in the browser, a Web Worker and Node.
export type RGB = { w: number; h: number; data: Uint8ClampedArray }; // RGBA, 4 bytes per pixel
export type Gray = { w: number; h: number; data: Uint8Array }; // 1 byte per pixel; masks use 0/1

export const gray = (w: number, h: number): Gray => ({ w, h, data: new Uint8Array(w * h) });

// HSV "value" = brightest channel. Larvae are bright on dark, so V separates them well.
export function toValue(img: RGB): Gray {
  const out = gray(img.w, img.h);
  for (let i = 0, j = 0; j < out.data.length; i += 4, j++)
    out.data[j] = Math.max(img.data[i], img.data[i + 1], img.data[i + 2]);
  return out;
}

const K = [1, 4, 6, 4, 1]; // binomial ≈ Gaussian, sums to 16

// Separable blur: rows then columns (2×5 multiplies per pixel instead of 25). Edges clamp.
export function gaussianBlur(g: Gray): Gray {
  const { w, h } = g, tmp = new Float32Array(w * h), out = gray(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += K[k + 2] * g.data[y * w + Math.min(w - 1, Math.max(0, x + k))];
      tmp[y * w + x] = s / 16;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += K[k + 2] * tmp[Math.min(h - 1, Math.max(0, y + k)) * w + x];
      out.data[y * w + x] = Math.round(s / 16);
    }
  return out;
}

// Each output pixel samples the input at its centre and mixes the 4 nearest pixels by distance.
export function resizeBilinear(img: RGB, w: number, h: number): RGB {
  const out = new Uint8ClampedArray(w * h * 4), sx = img.w / w, sy = img.h / h;
  const at = (x: number, y: number, c: number) => img.data[(y * img.w + x) * 4 + c];
  for (let y = 0; y < h; y++) {
    const fy = Math.min(img.h - 1, Math.max(0, (y + 0.5) * sy - 0.5)), y0 = Math.floor(fy);
    const y1 = Math.min(img.h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(img.w - 1, Math.max(0, (x + 0.5) * sx - 0.5)), x0 = Math.floor(fx);
      const x1 = Math.min(img.w - 1, x0 + 1), tx = fx - x0;
      for (let c = 0; c < 4; c++)
        out[(y * w + x) * 4 + c] =
          (at(x0, y0, c) * (1 - tx) + at(x1, y0, c) * tx) * (1 - ty) +
          (at(x0, y1, c) * (1 - tx) + at(x1, y1, c) * tx) * ty;
    }
  }
  return { w, h, data: out };
}
