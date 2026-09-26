// #/review — check SAM's draft masks ONE FLY AT A TIME. Talks to ml/draft_server.py on localhost:5055.
// Only approved flies become training data, so this page is where ground truth is made.
// Two ways to fix a part:
//   SAM click     click spots on the part, press Make <part>; Smaller / Bigger step through SAM's outlines
//   Draw outline  click the corners of a shape around the region, press Make <part> (or Erase) to fill it
import { useCallback, useEffect, useMemo, useState } from 'react';
import { misorderedFlies, type RGB } from '@flymorph/core';
import { decode } from './decode';
import { Overlay } from './Overlay';

const API = 'http://127.0.0.1:5055';
const TARGET = 60;
export const PART_COLORS: Record<number, [number, number, number]> = { 1: [255, 60, 60], 2: [40, 120, 255], 3: [80, 220, 80] };
const PART_NAMES = ['background', 'head', 'thorax', 'abdomen'];
const color = (l: number) => PART_COLORS[l] ?? null;

type Box = [number, number, number, number]; // x0, y0, x1, y1 in photo pixels
type Item = { sha: string; fly: number; flies: number; box: Box; session: string };
type Point = [number, number, 0 | 1];
type Tool = 'sam' | 'draw';
type Kind = 'approve' | 'reject' | 'flip' | 'refine' | 'paint' | 'reset' | 'undo' | 'skip';

const post = (path: string, body: object) =>
  fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

async function loadLabels(sha: string): Promise<Uint8Array> {
  const { data } = await decode(await (await fetch(`${API}/draft/${sha}?t=${Date.now()}`)).blob());
  return Uint8Array.from({ length: data.length / 4 }, (_, i) => data[i * 4]); // gray PNG → R channel = label
}

// Cut one fly's box out of the full photo and label map, so the page shows (and edits) a single fly.
function crop(img: RGB, labels: Uint8Array, [x0, y0, x1, y1]: Box) {
  const w = x1 - x0, h = y1 - y0, data = new Uint8ClampedArray(w * h * 4), lab = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    data.set(img.data.subarray(((y + y0) * img.w + x0) * 4, ((y + y0) * img.w + x1) * 4), y * w * 4);
    lab.set(labels.subarray((y + y0) * img.w + x0, (y + y0) * img.w + x1), y * w);
  }
  return { img: { w, h, data } as RGB, labels: lab };
}

export function Review() {
  const [queue, setQueue] = useState<Item[]>([]);
  const [counts, setCounts] = useState({ approved: 0, rejected: 0 });
  const [photo, setPhoto] = useState<RGB | null>(null);
  const [labels, setLabels] = useState<Uint8Array | null>(null);
  const [tool, setTool] = useState<Tool>('sam');
  const [points, setPoints] = useState<Point[]>([]); // photo coordinates
  const [polygon, setPolygon] = useState<[number, number][]>([]); // photo coordinates
  const [status, setStatus] = useState('');
  const [confirmBad, setConfirmBad] = useState(false); // approving a fly that fails the anatomy check needs 2 presses
  const [approved, setApproved] = useState<Item[]>([]); // this session's approvals, newest last, for Un-approve
  const [lastFix, setLastFix] = useState<{ part: number; pick: number; pts: Point[] } | null>(null); // Smaller / Bigger
  const current = queue[0];

  const refresh = useCallback(async (first?: Item) => {
    try {
      const q = await (await fetch(`${API}/api/queue`)).json();
      const same = (i: Item) => first && i.sha === first.sha && i.fly === first.fly;
      setQueue(first ? [...q.queue.filter(same), ...q.queue.filter((i: Item) => !same(i))] : q.queue);
      setCounts(q); setStatus('');
    } catch {
      setStatus('Draft server not running. Start it with: ml/.venv/bin/python ml/draft_server.py');
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  // take back the most recent approval and show that fly again
  const unapprove = useCallback(async () => {
    const last = approved[approved.length - 1];
    if (!last) { setStatus('No approvals to take back in this session.'); return; }
    await post('/api/unapprove', { sha: last.sha, fly: last.fly });
    setApproved((a) => a.slice(0, -1));
    await refresh(last);
    setStatus(`Un-approved fly ${last.fly + 1} of that photo. It's back on screen: fix it or reject it.`);
  }, [approved, refresh]);

  // the photo only reloads when the photo changes; moving to its next fly keeps it
  useEffect(() => {
    if (!current) return;
    setPhoto(null); setLabels(null);
    (async () => {
      setPhoto(await decode(await (await fetch(`${API}/img/${current.sha}`)).blob()));
      setLabels(await loadLabels(current.sha));
    })();
  }, [current?.sha]);
  useEffect(() => { setPoints([]); setPolygon([]); setLastFix(null); setConfirmBad(false); }, [current?.sha, current?.fly]);

  const view = useMemo(() => (photo && labels && current ? crop(photo, labels, current.box) : null), [photo, labels, current]);
  const bad = useMemo(() => (view ? misorderedFlies(view.labels, view.img.w, view.img.h) : []), [view]);

  const act = useCallback(async (kind: Kind, part = 0, pick = 0, pts: Point[] = points) => {
    if (!current) return;
    if (kind === 'skip') { setQueue((q) => [...q.slice(1), q[0]]); return; }
    if (kind === 'refine' && !pts.length) { setStatus('First click spots on the part, then choose what they are.'); return; }
    if (kind === 'paint' && polygon.length < 3) { setStatus('Click at least 3 corners around the region first.'); return; }
    if (kind === 'approve' && bad.length && !confirmBad) {
      setConfirmBad(true);
      setStatus('This fly has its head and abdomen on the same side of the thorax (✕), so a part is probably wrong. Fix it, or press Approve again to approve anyway.');
      return;
    }
    setStatus(kind === 'approve' || kind === 'reject' || kind === 'undo' || kind === 'paint' ? '' : 'SAM is thinking…');
    const res = await post(`/api/${kind}`, { sha: current.sha, fly: current.fly, box: current.box, part, points: pts, pick, polygon });
    if (!res.ok) { setStatus(kind === 'undo' ? 'Nothing to undo on this photo.' : `Server said no (${res.status}).`); return; }
    if (kind === 'approve') setApproved((a) => [...a, current]);
    if (kind === 'approve' || kind === 'reject') { refresh(); return; }
    setLabels(await loadLabels(current.sha)); setStatus(''); setConfirmBad(false);
    if (kind === 'refine') setLastFix({ part, pick, pts });
    if (kind === 'paint') setPolygon([]);
    if (kind === 'reset' || kind === 'flip') { setPoints([]); setPolygon([]); }
  }, [current, points, polygon, refresh, bad, confirmBad]);

  const make = useCallback((part: number) => (tool === 'draw' ? act('paint', part) : part ? act('refine', part) : undefined), [tool, act]);
  const resize = useCallback((d: number) => {
    if (lastFix && lastFix.pick + d >= 0 && lastFix.pick + d <= 2) act('refine', lastFix.part, lastFix.pick + d, lastFix.pts);
  }, [lastFix, act]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const map: Record<string, () => void> = {
        a: () => act('approve'), u: () => unapprove(), r: () => act('reject'), x: () => act('flip'), s: () => act('skip'), z: () => act('undo'),
        '0': () => make(0), '1': () => make(1), '2': () => make(2), '3': () => make(3),
        '-': () => resize(-1), '=': () => resize(1), '+': () => resize(1), d: () => setTool((t) => (t === 'sam' ? 'draw' : 'sam')),
        escape: () => { setPoints([]); setPolygon([]); },
      };
      map[e.key.toLowerCase()]?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [act, make, resize, unapprove]);

  if (!current) return <section><p>{status || 'Queue empty. Nice.'}</p></section>;
  const [bx, by] = current.box;
  const marks = [
    ...points.map(([x, y, pos]) => ({ x: x - bx - 5, y: y - by - 5, w: 10, h: 10, text: pos ? '+' : '−' })),
    ...bad.map(({ x, y }) => ({ x: x - 12, y: y - 12, w: 24, h: 24, text: '✕ check order' })),
  ];
  const clicked = tool === 'sam' ? points.length : polygon.length;

  return (
    <section className="review">
      <p>
        Photos done <b>{counts.approved}</b> / {TARGET} · rejected {counts.rejected} · {queue.length} flies left ·{' '}
        this photo: fly <b>{current.fly + 1}</b> of {current.flies} · <small>{current.session}</small>
      </p>
      <div className="controls">
        <button onClick={() => act('approve')}>Approve fly (A)</button>
        <button onClick={unapprove} disabled={!approved.length}>Un-approve last (U)</button>
        <button onClick={() => act('reject')}>Reject photo (R)</button>
        <button onClick={() => act('flip')}>Backwards (X)</button>
        <button onClick={() => act('skip')}>Skip (S)</button>
        <button onClick={() => act('undo')}>Undo (Z)</button>
        <button onClick={() => act('reset')}>Reset this fly</button>
      </div>
      <div className="controls">
        <span>Tool (D):</span>
        <button aria-pressed={tool === 'sam'} className={tool === 'sam' ? 'on' : ''} onClick={() => setTool('sam')}>SAM click</button>
        <button aria-pressed={tool === 'draw'} className={tool === 'draw' ? 'on' : ''} onClick={() => setTool('draw')}>Draw outline</button>
        <span>{tool === 'sam' ? `${points.length} spot(s) →` : `${polygon.length} corner(s) →`}</span>
        {[1, 2, 3].map((k) => (
          <button key={k} onClick={() => make(k)} disabled={tool === 'sam' ? !points.length : polygon.length < 3}
            style={{ borderBottom: `3px solid rgb(${PART_COLORS[k].join(',')})` }}>Make {PART_NAMES[k]} ({k})</button>
        ))}
        {tool === 'draw' && <button onClick={() => make(0)} disabled={polygon.length < 3}>Erase (0)</button>}
        {tool === 'sam' && <>
          <button onClick={() => resize(-1)} disabled={!lastFix || lastFix.pick === 0}>Smaller (−)</button>
          <button onClick={() => resize(1)} disabled={!lastFix || lastFix.pick === 2}>Bigger (+)</button>
        </>}
        <button onClick={() => { setPoints([]); setPolygon([]); }} disabled={!clicked}>Clear (Esc)</button>
      </div>
      <p className="keys">
        {tool === 'sam'
          ? 'SAM click: click on the part (shift-click = “not this”), then Make head / thorax / abdomen. Wrong size? Smaller / Bigger.'
          : 'Draw outline: click the corners of a shape around the region, then Make head / thorax / abdomen to fill it, or Erase to clear colour off legs and wings.'}
      </p>
      {status && <p className="error">{status}</p>}
      {view && (
        <Overlay img={view.img} labels={view.labels} color={color} boxes={marks}
          polygon={polygon.map(([x, y]) => [x - bx, y - by] as [number, number])}
          onClick={(x, y, e) => {
            const px = x + bx, py = y + by; // crop → photo coordinates
            if (tool === 'draw') setPolygon((p) => [...p, [px, py]]);
            else setPoints((p) => [...p, [px, py, e.shiftKey ? 0 : 1]]);
          }} />
      )}
    </section>
  );
}
