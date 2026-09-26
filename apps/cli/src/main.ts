#!/usr/bin/env -S npx tsx
// flymorph <larva|adult> <dir> --scale <scale.jpg> [--out results.csv]
// Measures every image in <dir> (files named *scale* are skipped) and writes one CSV row per object.
import { parseArgs } from 'node:util';
import { readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { calibrateFromScale, measureLarvae, toCSV, MIN_CONFIDENCE, type Measurement, type RGB, type Calibration } from '@flymorph/core';
import { decode } from './decode';

const USAGE = 'usage: flymorph <larva|adult> <dir> --scale <scale.jpg> [--out results.csv] [--px-per-mm N]';
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: { scale: { type: 'string' }, out: { type: 'string', default: 'results.csv' }, 'px-per-mm': { type: 'string' } },
});
const [mode, dir] = positionals;
if (!['larva', 'adult'].includes(mode) || !dir || !(values.scale || values['px-per-mm'])) {
  console.error(USAGE);
  process.exit(2);
}
if (mode === 'adult') {
  console.error('adult mode arrives in m09');
  process.exit(2);
}

let cal: Calibration;
if (values['px-per-mm']) cal = { pxPerMm: Number(values['px-per-mm']), confidence: Infinity, method: 'manual' };
else {
  cal = calibrateFromScale(await decode(values.scale!));
  if (cal.confidence < MIN_CONFIDENCE) {
    console.error(`scale bar not found in ${values.scale}; measure it by hand and pass --px-per-mm`);
    process.exit(1);
  }
}
console.error(`scale: ${cal.pxPerMm.toFixed(1)} px/mm (${cal.method})`);

const measure = async (img: RGB): Promise<Measurement[]> => measureLarvae(img, cal).measurements;

const files = (await readdir(dir)).filter((f) => /\.(jpe?g|png|tiff?)$/i.test(f) && !/scale/i.test(f)).sort();
const rows: Record<string, string | number | undefined>[] = [];
const failed: string[] = [];
for (const f of files) {
  try {
    const ms = await measure(await decode(join(dir, f)));
    for (const m of ms)
      rows.push({ file: f, id: m.id, part: m.part, area_mm2: m.areaMm2.toFixed(5), sd_mm2: m.sdMm2?.toFixed(5),
        area_px: m.areaPx, flag: m.flag, px_per_mm: cal.pxPerMm.toFixed(2) });
    console.error(`✓ ${f}: ${ms.length}`);
  } catch (e) {
    failed.push(f);
    console.error(`✗ ${f}: ${(e as Error).message}`);
  }
}
await writeFile(values.out!, toCSV(rows));
console.error(`${rows.length} rows → ${values.out}${failed.length ? `; ${failed.length} failed: ${failed.join(', ')}` : ''}`);
process.exit(failed.length ? 1 : 0);
