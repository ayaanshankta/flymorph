import { useState } from 'react';
import { labelsToImage, manualCalibration, measureAdult, toCSV, type Calibration, type Measurement, type RGB } from '@flymorph/core';
import { decode } from './decode';
import { loadModel } from './model';
import { Overlay, type Box } from './Overlay';
import { PART_COLORS } from './Review';
import type { WorkerRequest } from './worker';

export type Mode = 'larva' | 'adult';
type Photo = { name: string; rgb: RGB };
type Result = Photo & { measurements: Measurement[]; labels: ArrayLike<number> };

const LARVA_COLOR = (l: number): [number, number, number] | null => (l ? [57, 255, 136] : null);
const PART_COLOR = (l: number) => PART_COLORS[l] ?? null;
const COPY: Record<Mode, { thing: string; things: string; photo: string; how: string }> = {
  larva: {
    thing: 'larva', things: 'larvae', photo: 'Larva photos',
    how: 'Larvae on a dark background.',
  },
  adult: {
    thing: 'fly', things: 'flies', photo: 'Fly photos',
    how: 'Head, thorax and abdomen area per fly (legs and wings not counted).',
  },
};
const FLAGS: Record<string, string> = {
  edge: 'touches the photo edge, so part of it may be cut off',
  irregular: 'shape is not larva-like (touching another larva or an object?)',
  uncertain: 'the model is unsure about this fly (± above 10%)',
  'order?': 'head and abdomen are on the same side of the thorax',
  'gap?': 'gap between body parts: some body may be missing',
  'sizes?': 'head larger than thorax or abdomen',
  'pieces?': 'a body part is split in two',
};

function inWorker<T>(req: WorkerRequest): Promise<T> {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      worker.terminate();
      e.data.ok ? resolve(e.data) : reject(Object.assign(new Error(e.data.error), { needsScale: !!e.data.needsScale }));
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message)); };
    worker.postMessage(req);
  });
}

const fmt = (m: Measurement) => m.areaMm2.toFixed(3) + (m.sdMm2 !== undefined ? ` ± ${m.sdMm2.toFixed(3)}` : '');
const nextFrame = () => new Promise((r) => setTimeout(r, 30)); // let the progress text paint before heavy work

export function Measure({ mode }: { mode: Mode }) {
  const c = COPY[mode];
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [scale, setScale] = useState<Photo | null>(null);
  const [clicks, setClicks] = useState<[number, number][]>([]);
  const [cal, setCal] = useState<Calibration | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [error, setError] = useState('');
  const [askScale, setAskScale] = useState(false);
  const [busy, setBusy] = useState('');
  const [csvShown, setCsvShown] = useState(false);
  const [copied, setCopied] = useState(false);

  const run = async (ph: Photo[], sc: Photo | null, known?: Calibration) => {
    if (!ph.length) return setError(`Add at least one ${c.thing} photo.`);
    if (!sc && !known) return setError('Add the scale photo (the stage micrometer, taken at the same zoom).');
    setError(''); setAskScale(false); setResults([]); setCsvShown(false);
    try {
      setBusy('Reading the scale bar…'); await nextFrame();
      const calib = known ?? (await inWorker<{ cal: Calibration }>({ mode: 'calibrate', scale: sc!.rgb })).cal;
      setCal(calib);
      const model = mode === 'adult' ? (setBusy('Loading the model (first time only, ~20 MB)…'), await loadModel()) : null;
      const out: Result[] = [];
      for (const [i, p] of ph.entries()) {
        setBusy(`Measuring photo ${i + 1} of ${ph.length}…`); await nextFrame();
        if (model) {
          const r = await measureAdult(p.rgb, calib, model);
          out.push({ ...p, measurements: r.measurements, labels: labelsToImage(r.labels, r.box, p.rgb.w, p.rgb.h) });
        } else {
          const r = await inWorker<{ measurements: Measurement[]; labels: Int32Array }>({ mode: 'larva', img: p.rgb, cal: calib });
          out.push({ ...p, measurements: r.measurements, labels: r.labels });
        }
        setResults([...out]);
      }
    } catch (e) {
      setError((e as Error).message);
      setAskScale(!!(e as { needsScale?: boolean }).needsScale);
    } finally {
      setBusy('');
    }
  };

  const load = async (files: FileList | null, multiple: boolean) => {
    if (!files?.length) return null;
    setBusy('Loading photos…');
    try {
      const out: Photo[] = [];
      for (const f of Array.from(files).slice(0, multiple ? 50 : 1)) out.push({ name: f.name, rgb: await decode(f) });
      return out;
    } catch {
      setError('One of those files could not be read as an image.');
      return null;
    } finally {
      setBusy('');
    }
  };

  const example = async () => {
    setBusy('Loading the example…');
    const get = async (file: string) => ({ name: file, rgb: await decode(await (await fetch(`examples/${file}`)).blob()) });
    const [p, s] = await Promise.all([get(`${mode}.jpg`), get(`${mode}-scale.jpg`)]);
    setPhotos([p]); setScale(s); setClicks([]); setCal(null);
    await run([p], s);
  };

  const onScaleClick = (x: number, y: number) => {
    const next = [...clicks, [x, y] as [number, number]].slice(-2);
    setClicks(next);
    if (next.length === 2) run(photos, scale, manualCalibration(next[0], next[1]));
  };

  const rows = results.flatMap((r) => r.measurements.map((m) => ({
    photo: r.name, [c.thing]: m.id, ...(mode === 'adult' ? { part: m.part } : {}),
    area_mm2: m.areaMm2.toFixed(4), ...(mode === 'adult' ? { sd_mm2: m.sdMm2?.toFixed(4) } : {}),
    flag: m.flag ?? '', px_per_mm: cal?.pxPerMm.toFixed(1),
  })));
  const csv = toCSV(rows);
  const copy = async () => {
    try { await navigator.clipboard.writeText(csv); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch { setCsvShown(true); } // clipboard refused: show the text to copy by hand
  };
  const count = (r: Result) => r.measurements.filter((m) => m.part === 'larva' || m.part === 'body').length;
  const flagsSeen = [...new Set(results.flatMap((r) => r.measurements.flatMap((m) => (m.flag ? m.flag.split(',') : []))))];

  return (
    <section className="tool">
      <p className="how">{c.how}</p>

      <div className="inputs">
        <div className="field">
          <span className="field-label">{c.photo}</span>
          <label className="file" htmlFor={`${mode}-photos`}>
            <input id={`${mode}-photos`} className="visually-hidden" type="file" accept="image/*" multiple onChange={async (e) => {
              const p = await load(e.target.files, true); if (p) { setPhotos(p); setResults([]); }
            }} />
            <span className="file-btn">Choose…</span>
            <span className="file-name">{photos.length ? (photos.length === 1 ? photos[0].name : `${photos.length} photos`) : 'No photos chosen'}</span>
          </label>
        </div>
        <div className="field">
          <span className="field-label">Scale photo <span className="hint">same zoom</span></span>
          <label className="file" htmlFor={`${mode}-scale`}>
            <input id={`${mode}-scale`} className="visually-hidden" type="file" accept="image/*" onChange={async (e) => {
              const s = await load(e.target.files, false); if (s) { setScale(s[0]); setClicks([]); setCal(null); }
            }} />
            <span className="file-btn">Choose…</span>
            <span className="file-name">{scale ? scale.name : 'No scale photo'}</span>
          </label>
        </div>
        <div className="actions">
          <button className="primary" disabled={!!busy} onClick={() => run(photos, scale)}>Measure</button>
        </div>
      </div>

      {busy && <p className="status" role="status">{busy}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {askScale && scale && (
        <div className="panel">
          <p>Click the left end, then the right end of the 1 mm bar ({clicks.length}/2).</p>
          <Overlay img={scale.rgb} onClick={onScaleClick}
            boxes={clicks.map(([x, y], i) => ({ x: x - 6, y: y - 6, w: 12, h: 12, text: String(i + 1) }))} />
        </div>
      )}

      {!results.length && !busy && !error && (
        <div className="empty">
          <img src={`examples/${mode}.jpg`} alt="" />
          <div>
            <p className="empty-title">No photos measured yet</p>
            <p>Choose {c.things === 'larvae' ? 'larva' : 'fly'} photos and the stage-micrometer photo taken at the same zoom, then
              press Measure. Results come out as a table you can paste into Excel or Sheets.</p>
            <button onClick={example}>Try it on this example photo</button>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="results">
          <div className="summary">
            <span>
              {results.reduce((a, r) => a + count(r), 0)} {c.things} in {results.length} photo{results.length > 1 ? 's' : ''} ·
              scale {cal?.pxPerMm.toFixed(0)} px/mm{cal?.method === 'manual' ? ' (from your clicks)' : ''}
            </span>
            <button onClick={copy}>{copied ? 'Copied' : 'Copy table'}</button>
          </div>
          {csvShown && <textarea id={`${mode}-csv`} readOnly value={csv} onFocus={(e) => e.target.select()} rows={6} />}

          <div className="scroll">
            <table>
              <thead><tr><th>Photo</th><th>#</th>{mode === 'adult' && <th>Part</th>}<th className="num">Area (mm²)</th><th>Flag</th></tr></thead>
              <tbody>{results.flatMap((r) => r.measurements.map((m) => (
                <tr key={`${r.name}-${m.id}-${m.part}`} className={m.part === 'body' ? 'total' : ''}>
                  <td>{r.name}</td><td>{m.id}</td>{mode === 'adult' && <td>{m.part}</td>}
                  <td className="num">{fmt(m)}</td><td className="flag">{m.flag ?? ''}</td>
                </tr>
              )))}</tbody>
            </table>
          </div>
          {flagsSeen.length > 0 && (
            <dl className="flags">{flagsSeen.map((f) => <div key={f}><dt>{f}</dt><dd>{FLAGS[f] ?? ''}</dd></div>)}</dl>
          )}

          {mode === 'adult' && (
            <div className="legend">{[1, 2, 3].map((k) => (
              <span key={k}><i style={{ background: `rgb(${PART_COLORS[k].join(',')})` }} />{['', 'head', 'thorax', 'abdomen'][k]}</span>
            ))}</div>
          )}
          <div className="figures">{results.map((r) => (
            <figure key={r.name}>
              <Overlay img={r.rgb} labels={r.labels} color={mode === 'larva' ? LARVA_COLOR : PART_COLOR}
                boxes={r.measurements.filter((m) => m.part === 'larva' || m.part === 'body').map((m): Box => ({
                  x: m.bbox[0], y: m.bbox[1], w: m.bbox[2], h: m.bbox[3], text: `${m.id}: ${m.areaMm2.toFixed(2)} mm²`,
                }))} />
              <figcaption>{r.name} · {count(r) ? `${count(r)} ${count(r) > 1 ? c.things : c.thing}` : `no ${c.things} found`}</figcaption>
            </figure>
          ))}</div>
        </div>
      )}
    </section>
  );
}
