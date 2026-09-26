import sharp from 'sharp';
import type { RGB } from '@flymorph/core';

// sharp decodes JPEG/PNG/TIFF; .rotate() applies EXIF orientation, .ensureAlpha() makes every input RGBA,
// matching what the browser's canvas gives the web app.
export async function decode(path: string): Promise<RGB> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { w: info.width, h: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
}
