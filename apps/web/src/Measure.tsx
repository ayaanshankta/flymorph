import { useMemo, useState } from 'react';
import { manualCalibration, toCSV, type Calibration, type Measurement, type RGB } from '@flymorph/core';
import { decode } from './decode';
import { Overlay, type Box } from './Overlay';
import type { WorkerRequest } from './worker';

type Result = { cal: Calibration; measurements: Measurement[]; labels: ArrayLike<number> };

const LARVA_COLOR = (l: number): [number, number, number] | null => (l ? [57, 255, 136] : null);

function run(req: WorkerRequest): Promise<Result> {
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

function download(name: string, text: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function Measure() {
  const [img, setImg] = useState<{ name: string; rgb: RGB } | null>(null);
  const [scale, setScale] = useState<RGB | null>(null);
  const [clicks, setClicks] = useState<[number, number][]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const measure = async (manualPxPerMm?: number) => {
    if (!img) return;
    setBusy(true); setError(''); setResult(null);
    try { setResult(await run({ img: img.rgb, scale, manualPxPerMm })); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const onScaleClick = (x: number, y: number) => {
    const next = [...clicks, [x, y] as [number, number]].slice(-2);
    setClicks(next);
    if (next.length === 2) measure(manualCalibration(next[0], next[1]).pxPerMm);
  };

  const boxes: Box[] = useMemo(() => (result?.measurements ?? []).map((m) => ({
    x: m.bbox[0], y: m.bbox[1], w: m.bbox[2], h: m.bbox[3],
    text: `${m.id}: ${m.areaMm2.toFixed(3)} mm²${m.flag ? ` (${m.flag})` : ''}`,
  })), [result]);

  const csv = () => img && result && download(img.name.replace(/\.\w+$/, '') + '_larvae.csv', toCSV(result.measurements.map((m) => ({
    file: img.name, id: m.id, area_mm2: m.areaMm2.toFixed(5), area_px: m.areaPx, flag: m.flag,
    px_per_mm: result.cal.pxPerMm.toFixed(2), calibration: result.cal.method,
  }))));

  return (
    <section>
      <div className="controls">
        <label>Larva image <input type="file" accept="image/*" onChange={async (e) => {
          const f = e.target.files?.[0]; if (f) { setImg({ name: f.name, rgb: await decode(f) }); setResult(null); }
        }} /></label>
        <label>Scale image <input type="file" accept="image/*" onChange={async (e) => {
          const f = e.target.files?.[0]; if (f) { setScale(await decode(f)); setClicks([]); }
        }} /></label>
        <button disabled={!img || busy} onClick={() => measure()}>{busy ? 'Measuring…' : 'Measure'}</button>
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
            {result.measurements.length ? `${result.measurements.length} larvae` : 'No larvae found'}
            {result.measurements.length > 0 && <> · <button onClick={csv}>Download CSV</button></>}
          </p>
          <table>
            <thead><tr><th>#</th><th>Area (mm²)</th><th>Pixels</th><th>Flag</th></tr></thead>
            <tbody>{result.measurements.map((m) => (
              <tr key={m.id}><td>{m.id}</td><td>{m.areaMm2.toFixed(4)}</td><td>{m.areaPx}</td><td>{m.flag ?? ''}</td></tr>
            ))}</tbody>
          </table>
        </>
      )}

      {img && <Overlay img={img.rgb} labels={result?.labels} color={LARVA_COLOR} boxes={boxes} />}
    </section>
  );
}
