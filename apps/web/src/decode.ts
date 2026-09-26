import type { RGB } from '@flymorph/core';

// createImageBitmap applies EXIF rotation; a canvas always hands back RGBA,
// so grayscale JPEGs and PNGs with alpha come out in the same format.
export async function decode(file: Blob): Promise<RGB> {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  return { w: bmp.width, h: bmp.height, data: ctx.getImageData(0, 0, bmp.width, bmp.height).data };
}
