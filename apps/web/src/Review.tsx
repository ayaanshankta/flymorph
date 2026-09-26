// #/review — approve, reject or fix SAM's draft masks. Talks to ml/draft_server.py on localhost:5055.
// Only masks approved here become training data, so this page is where ground truth is made.
import { useCallback, useEffect, useState } from 'react';
import type { RGB } from '@flymorph/core';
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
  const [part, setPart] = useState(1);
  const [points, setPoints] = useState<Point[]>([]);
  const [status, setStatus] = useState('');
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
    setImg(null); setLabels(null); setPoints([]);
    (async () => {
      setImg(await decode(await (await fetch(`${API}/img/${current.sha}`)).blob()));
      setLabels(await loadLabels(current.sha));
    })();
  }, [current?.sha]);

  const act = useCallback(async (kind: 'approve' | 'reject' | 'flip' | 'refine' | 'skip') => {
    if (!current) return;
    if (kind === 'skip') { setQueue((q) => [...q.slice(1), q[0]]); return; }
    if (kind === 'refine' && !points.length) { setStatus('Click on the part first (shift-click = "not this").'); return; }
    setStatus(kind === 'flip' || kind === 'refine' ? 'SAM is thinking…' : '');
    await post(`/api/${kind}`, { sha: current.sha, part, points });
    if (kind === 'flip' || kind === 'refine') { setLabels(await loadLabels(current.sha)); setPoints([]); setStatus(''); }
    else refresh();
  }, [current, part, points, refresh]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      const map: Record<string, () => void> = {
        a: () => act('approve'), r: () => act('reject'), x: () => act('flip'), f: () => act('refine'), s: () => act('skip'),
        '1': () => setPart(1), '2': () => setPart(2), '3': () => setPart(3), escape: () => setPoints([]),
      };
      map[k]?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [act]);

  const boxes = points.map(([x, y, pos]) => ({ x: x - 5, y: y - 5, w: 10, h: 10, text: pos ? '+' : '−' }));

  return (
    <section>
      <p>
        Approved <b>{counts.approved}</b> / {TARGET} · rejected {counts.rejected} · {queue.length} left
        {current && <> · <small>{current.session}</small></>}
      </p>
      <p className="keys">
        <b>A</b> approve · <b>R</b> reject · <b>X</b> fly is backwards · <b>S</b> skip ·{' '}
        fix a part: <b>1</b> head <b>2</b> thorax <b>3</b> abdomen, click it (shift-click = not this), <b>F</b> apply ·{' '}
        editing: <span style={{ color: `rgb(${PART_COLORS[part].join(',')})` }}>{PART_NAMES[part]}</span>
      </p>
      {status && <p className="error">{status}</p>}
      {!current && !status && <p>Queue empty. Nice.</p>}
      {img && labels && (
        <Overlay img={img} labels={labels} color={color} boxes={boxes}
          onClick={(x, y, e) => setPoints((p) => [...p, [x, y, e.shiftKey ? 0 : 1]])} />
      )}
    </section>
  );
}
