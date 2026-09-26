// #/review — approve, reject or fix SAM's draft masks. Talks to ml/draft_server.py on localhost:5055.
// Only masks approved here become training data, so this page is where ground truth is made.
import { useCallback, useEffect, useMemo, useState } from 'react';
import { misorderedFlies, type RGB } from '@flymorph/core';
import { decode } from './decode';
import { Overlay } from './Overlay';

const API = 'http://127.0.0.1:5055';
const TARGET = 60;
export const PART_COLORS: Record<number, [number, number, number]> = { 1: [255, 60, 60], 2: [40, 120, 255], 3: [80, 220, 80] };
const PART_NAMES = ['', 'head', 'thorax', 'abdomen'];
const color = (l: number) => PART_COLORS[l] ?? null;

type Item = { sha: string; session: string };
type Point = [number, number, 0 | 1];

const post = (path: string, body: object) =>
  fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

async function loadLabels(sha: string): Promise<Uint8Array> {
  const { data } = await decode(await (await fetch(`${API}/draft/${sha}?t=${Date.now()}`)).blob());
  return Uint8Array.from({ length: data.length / 4 }, (_, i) => data[i * 4]); // gray PNG → R channel = label
}

export function Review() {
  const [queue, setQueue] = useState<Item[]>([]);
  const [counts, setCounts] = useState({ approved: 0, rejected: 0 });
  const [img, setImg] = useState<RGB | null>(null);
  const [labels, setLabels] = useState<Uint8Array | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [status, setStatus] = useState('');
  const [confirmBad, setConfirmBad] = useState(false); // approving a fly that fails the anatomy check needs 2 presses
  const [lastFix, setLastFix] = useState<{ part: number; pick: number } | null>(null); // for Smaller / Bigger
  const current = queue[0];

  const refresh = useCallback(async () => {
    try {
      const q = await (await fetch(`${API}/api/queue`)).json();
      setQueue(q.queue); setCounts(q); setStatus('');
    } catch {
      setStatus('Draft server not running. Start it with: ml/.venv/bin/python ml/draft_server.py');
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!current) return;
    setImg(null); setLabels(null); setPoints([]); setLastFix(null); setConfirmBad(false);
    (async () => {
      setImg(await decode(await (await fetch(`${API}/img/${current.sha}`)).blob()));
      setLabels(await loadLabels(current.sha));
    })();
  }, [current?.sha]);

  // part is only used by 'refine': the fix names its part at the moment it is applied (no hidden mode).
  const act = useCallback(async (kind: 'approve' | 'reject' | 'flip' | 'refine' | 'reset' | 'undo' | 'skip', part = 0, pick = 0) => {
    if (!current) return;
    if (kind === 'skip') { setQueue((q) => [...q.slice(1), q[0]]); return; }
    if (kind === 'approve' && img && labels && misorderedFlies(labels, img.w, img.h).length && !confirmBad) {
      setConfirmBad(true);
      setStatus('A fly here has its head and abdomen on the same side of the thorax (marked ✕), so a part is probably mislabelled. Fix it, or press Approve again to approve anyway.');
      return;
    }
    if (kind === 'refine' && !points.length) { setStatus('First click the spots on the fly, then choose what they are.'); return; }
    setStatus(kind === 'approve' || kind === 'reject' || kind === 'undo' ? '' : 'SAM is thinking…');
    const res = await post(`/api/${kind}`, { sha: current.sha, part, points, pick });
    if (kind === 'undo' && !res.ok) { setStatus('Nothing to undo on this fly.'); return; }
    if (kind === 'flip' || kind === 'refine' || kind === 'reset' || kind === 'undo') {
      setLabels(await loadLabels(current.sha)); setStatus(''); setConfirmBad(false);
      // keep the clicks after a fix so Smaller / Bigger can re-ask SAM with the same spots
      if (kind === 'refine') setLastFix({ part, pick }); else { setPoints([]); setLastFix(null); }
    }
    else refresh();
  }, [current, points, refresh, img, labels, confirmBad]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      const map: Record<string, () => void> = {
        z: () => act('undo'),
        a: () => act('approve'), r: () => act('reject'), x: () => act('flip'), s: () => act('skip'),
        '1': () => act('refine', 1), '2': () => act('refine', 2), '3': () => act('refine', 3), escape: () => setPoints([]),
      };
      map[k]?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [act]);

  const bad = useMemo(() => (img && labels ? misorderedFlies(labels, img.w, img.h) : []), [img, labels]);
  const boxes = [
    ...points.map(([x, y, pos]) => ({ x: x - 5, y: y - 5, w: 10, h: 10, text: pos ? '+' : '−' })),
    ...bad.map(({ x, y }) => ({ x: x - 12, y: y - 12, w: 24, h: 24, text: '✕ check order' })),
  ];

  return (
    <section>
      <p>
        Approved <b>{counts.approved}</b> / {TARGET} · rejected {counts.rejected} · {queue.length} left
        {current && <> · <small>{current.session}</small></>}
      </p>
      <p className="keys">
        Looks right → <b>Approve</b>. Wrong part somewhere → click on that spot (shift-click = “not this”), then press
        what it should be: <b>Make head / thorax / abdomen</b>. Messed it up → <b>Undo</b> (last step) or <b>Reset</b> (start over).
      </p>
      {current && (
        <div className="controls">
          <button onClick={() => act('approve')}>Approve (A)</button>
          <button onClick={() => act('reject')}>Reject (R)</button>
          <button onClick={() => act('flip')}>Backwards (X)</button>
          <button onClick={() => act('skip')}>Skip (S)</button>
          <span>{points.length ? `${points.length} spot${points.length > 1 ? 's' : ''} clicked → make it:` : 'Click a spot to fix it'}</span>
          {[1, 2, 3].map((k) => (
            <button key={k} onClick={() => act('refine', k)} disabled={!points.length}
              style={{ borderBottom: `3px solid rgb(${PART_COLORS[k].join(',')})` }}>
              Make {PART_NAMES[k]} ({k})
            </button>
          ))}
          {lastFix && <>
            <button onClick={() => act('refine', lastFix.part, lastFix.pick - 1)} disabled={lastFix.pick === 0}>Smaller</button>
            <button onClick={() => act('refine', lastFix.part, lastFix.pick + 1)} disabled={lastFix.pick === 2}>Bigger</button>
          </>}
          <button onClick={() => { setPoints([]); setLastFix(null); }} disabled={!points.length}>Clear clicks</button>
          <button onClick={() => act('undo')}>Undo (Z)</button>
          <button onClick={() => act('reset')}>Reset to SAM draft</button>
        </div>
      )}
      {status && <p className="error">{status}</p>}
      {!current && !status && <p>Queue empty. Nice.</p>}
      {img && labels && (
        <Overlay img={img} labels={labels} color={color} boxes={boxes}
          onClick={(x, y, e) => { setLastFix(null); setPoints((p) => [...p, [x, y, e.shiftKey ? 0 : 1]]); }} />
      )}
    </section>
  );
}
