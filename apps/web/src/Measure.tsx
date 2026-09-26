import { useMemo, useState } from 'react';
import { labelsToImage, manualCalibration, measureAdult, toCSV, type Calibration, type Measurement, type RGB } from '@flymorph/core';
import { decode } from './decode';
import { loadModel } from './model';
import { Overlay, type Box } from './Overlay';
import { PART_COLORS } from './Review';
import type { WorkerRequest } from './worker';

type Mode = 'larva' | 'adult';
type Result = { cal: Calibration; measurements: Measurement[]; labels: ArrayLike<number>; note?: string };

const LARVA_COLOR = (l: number): [number, number, number] | null => (l ? [57, 255, 136] : null);
const PART_COLOR = (l: number) => PART_COLORS[l] ?? null;

function inWorker(req: WorkerRequest): Promise<Result> {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      worker.terminate();
      e.data.ok ? resolve(e.data) : reject(new Error(e.data.error));
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message)); };
    worker.postMessage(req);
  });
}

async function run(mode: Mode, img: RGB, scale: RGB | null, manualPxPerMm?: number): Promise<Result> {
  if (mode === 'larva') return inWorker({ mode, img, scale, manualPxPerMm });
  const [{ cal }, model] = await Promise.all([inWorker({ mode: 'calibrate', img, scale, manualPxPerMm }), loadModel()]);
  const r = await measureAdult(img, cal, model.run);
  return { cal, measurements: r.measurements, labels: labelsToImage(r.labels, r.box, img.w, img.h), note: `model on ${model.backend}` };
}

function download(name: string, text: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

const fmt = (m: Measurement) => m.areaMm2.toFixed(4) + (m.sdMm2 !== undefined ? ` ± ${m.sdMm2.toFixed(4)}` : '');

export function Measure() {
  const [mode, setMode] = useState<Mode>('larva');
  const [img, setImg] = useState<{ name: string; rgb: RGB } | null>(null);
  const [scale, setScale] = useState<RGB | null>(null);
  const [clicks, setClicks] = useState<[number, number][]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');

  const measure = async (manualPxPerMm?: number) => {
    if (!img) return;
    setBusy('Measuring…'); setError(''); setResult(null);
    try { setResult(await run(mode, img.rgb, scale, manualPxPerMm)); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(''); }
  };

  const load = async (f: File | undefined, set: (r: RGB) => void) => {
    if (!f) return;
    setBusy('Loading image…');
    try { set(await decode(f)); } catch { setError(`Could not read ${f.name} as an image.`); } finally { setBusy(''); }
  };

  const onScaleClick = (x: number, y: number) => {
    const next = [...clicks, [x, y] as [number, number]].slice(-2);
    setClicks(next);
    if (next.length === 2) measure(manualCalibration(next[0], next[1]).pxPerMm);
  };

  // one box per larva / per fly (adult rows share the fly's bbox; label it with the body area)
  const boxes: Box[] = useMemo(() => (result?.measurements ?? [])
    .filter((m) => m.part === 'larva' || m.part === 'body')
    .map((m) => ({ x: m.bbox[0], y: m.bbox[1], w: m.bbox[2], h: m.bbox[3],
      text: `${m.id}: ${fmt(m)} mm²${m.flag ? ` (${m.flag})` : ''}` })), [result]);

  const csv = () => img && result && download(`${img.name.replace(/\.\w+$/, '')}_${mode}.csv`, toCSV(result.measurements.map((m) => ({
    file: img.name, id: m.id, part: m.part, area_mm2: m.areaMm2.toFixed(5), sd_mm2: m.sdMm2?.toFixed(5), area_px: m.areaPx,
    flag: m.flag, px_per_mm: result.cal.pxPerMm.toFixed(2), calibration: result.cal.method,
  }))));

  const count = result?.measurements.filter((m) => m.part === 'larva' || m.part === 'body').length ?? 0;

  return (
    <section>
      <div className="controls">
        <select value={mode} onChange={(e) => { setMode(e.target.value as Mode); setResult(null); }}>
          <option value="larva">Larvae (image processing)</option>
          <option value="adult">Adult flies (ML model)</option>
        </select>
        <label>Image <input type="file" accept="image/*" onChange={(e) => { setResult(null); load(e.target.files?.[0], (rgb) => setImg({ name: e.target.files![0].name, rgb })); }} /></label>
        <label>Scale image <input type="file" accept="image/*" onChange={(e) => { setClicks([]); load(e.target.files?.[0], setScale); }} /></label>
        <button disabled={!img || !!busy} onClick={() => measure()}>{busy || 'Measure'}</button>
      </div>

      {error && <p className="error">{error}</p>}
      {error && scale && (
        <>
          <p>Click the left end, then the right end of the 1 mm bar ({clicks.length}/2):</p>
          <Overlay img={scale} onClick={onScaleClick}
            boxes={clicks.map(([x, y], i) => ({ x: x - 6, y: y - 6, w: 12, h: 12, text: String(i + 1) }))} />
        </>
      )}

      {result && (
        <>
          <p>
            Scale {result.cal.pxPerMm.toFixed(1)} px/mm ({result.cal.method}
            {result.cal.method === 'fft' ? `, confidence ${result.cal.confidence.toFixed(0)}` : ''}) ·{' '}
            {count ? `${count} ${mode === 'larva' ? 'larvae' : 'flies'}` : `No ${mode === 'larva' ? 'larvae' : 'flies'} found`}
            {result.note && <> · {result.note}</>}
            {count > 0 && <> · <button onClick={csv}>Download CSV</button></>}
          </p>
          <table>
            <thead><tr><th>#</th><th>Part</th><th>Area (mm²)</th><th>Pixels</th><th>Flag</th></tr></thead>
            <tbody>{result.measurements.map((m) => (
              <tr key={`${m.id}-${m.part}`}><td>{m.id}</td><td>{m.part}</td><td>{fmt(m)}</td><td>{m.areaPx}</td><td>{m.flag ?? ''}</td></tr>
            ))}</tbody>
          </table>
        </>
      )}

      {img && <Overlay img={img.rgb} labels={result?.labels} color={mode === 'larva' ? LARVA_COLOR : PART_COLOR} boxes={boxes} />}
    </section>
  );
}
