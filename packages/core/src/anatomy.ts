import { components } from './ccl';
import { dilate } from './morph';
import { gray } from './image';

// Simple body-plan rules that catch most bad masks (labels: 1 head, 2 thorax, 3 abdomen):
//   order   the thorax sits BETWEEN head and abdomen: seen from its centre they point opposite ways
//   gap     head and abdomen both touch (or nearly touch) the thorax; a wide gap = body the mask missed
//   sizes   the head is the smallest part
//   pieces  a part is split into two big pieces (a 2nd piece ≥ 40% of the part). Legs crossing the body
//           chip off smaller bits in real photos: on 154 approved flies the 2nd piece never passed 33%
export type Issue = 'order' | 'gap' | 'sizes' | 'pieces';
export type FlyIssues = { x: number; y: number; issues: Issue[] }; // x, y = thorax centre

// Which pixels form one fly: either a radius (mask pieces closer than 2·radius count as one fly, so a fly
// with a gap is still judged as one) or a ready-made map of fly ids (0 = not a fly), e.g. from postprocess.
export function anatomyIssues(labels: ArrayLike<number>, w: number, h: number,
  grouping: number | Int32Array = Math.max(1, Math.round(0.08 * Math.max(w, h)))): FlyIssues[] {
  let group: Int32Array;
  if (typeof grouping === 'number') {
    const any = gray(w, h);
    for (let i = 0; i < w * h; i++) any.data[i] = labels[i] ? 1 : 0;
    group = components(dilate(any, grouping)).labels;
  } else group = grouping;
  const out: FlyIssues[] = [];
  for (const c of boxesOf(group, w)) {
    const n = [0, 0, 0, 0], sx = [0, 0, 0, 0], sy = [0, 0, 0, 0];
    const part = (k: number) => {
      const m = gray(c.w, c.h);
      for (let y = 0; y < c.h; y++)
        for (let x = 0; x < c.w; x++) {
          const i = (y + c.y) * w + x + c.x;
          if (group[i] === c.id && labels[i] === k) m.data[y * c.w + x] = 1;
        }
      return m;
    };
    const masks = [gray(0, 0), part(1), part(2), part(3)];
    for (let k = 1; k <= 3; k++)
      masks[k].data.forEach((v, i) => { if (v) { n[k]++; sx[k] += i % c.w; sy[k] += Math.floor(i / c.w); } });
    if (!n[1] || !n[2] || !n[3]) continue; // can't judge a fly with a missing part
    const [hx, hy, tx, ty, ax, ay] = [sx[1] / n[1], sy[1] / n[1], sx[2] / n[2], sy[2] / n[2], sx[3] / n[3], sy[3] / n[3]];
    const issues: Issue[] = [];
    if ((hx - tx) * (ax - tx) + (hy - ty) * (ay - ty) >= 0) issues.push('order');
    const tooFar = Math.max(3, 0.08 * Math.max(c.w, c.h)), toThorax = distanceTo(masks[2]);
    const gapTo = (m: typeof masks[number]) => m.data.reduce((best, v, i) => (v ? Math.min(best, toThorax[i]) : best), Infinity);
    if (gapTo(masks[1]) > tooFar || gapTo(masks[3]) > tooFar) issues.push('gap');
    if (n[1] > n[2] || n[1] > n[3]) issues.push('sizes');
    if ([1, 2, 3].some((k) => components(masks[k]).comps.filter((p) => p.area >= 0.4 * n[k]).length > 1)) issues.push('pieces');
    if (issues.length) out.push({ x: tx + c.x, y: ty + c.y, issues });
  }
  return out;
}

function boxesOf(group: Int32Array, w: number) {
  const b = new Map<number, { id: number; x: number; y: number; w: number; h: number; x1: number; y1: number }>();
  for (let i = 0; i < group.length; i++) {
    const id = group[i];
    if (!id) continue;
    const x = i % w, y = (i - x) / w, c = b.get(id);
    if (!c) b.set(id, { id, x, y, w: 1, h: 1, x1: x, y1: y });
    else { c.x = Math.min(c.x, x); c.y = Math.min(c.y, y); c.x1 = Math.max(c.x1, x); c.y1 = Math.max(c.y1, y); }
  }
  return [...b.values()].map((c) => ({ ...c, w: c.x1 - c.x + 1, h: c.y1 - c.y + 1 }));
}

// Flies whose thorax is not between head and abdomen (kept for callers that only need the order rule).
export function misorderedFlies(labels: ArrayLike<number>, w: number, h: number) {
  return anatomyIssues(labels, w, h).filter((f) => f.issues.includes('order'));
}

// Chamfer distance transform: approximate pixel distance from every pixel to the nearest pixel of `m`.
// Two sweeps (top-left → bottom-right, then back) with cost 1 for straight and 1.4 for diagonal steps.
function distanceTo(m: { w: number; h: number; data: Uint8Array }): Float32Array {
  const { w, h } = m, d = new Float32Array(w * h).map((_, i) => (m.data[i] ? 0 : Infinity));
  const relax = (i: number, j: number, cost: number) => { if (d[j] + cost < d[i]) d[i] = d[j] + cost; };
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x > 0) relax(i, i - 1, 1);
      if (y > 0) { relax(i, i - w, 1); if (x > 0) relax(i, i - w - 1, 1.4); if (x < w - 1) relax(i, i - w + 1, 1.4); }
    }
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (x < w - 1) relax(i, i + 1, 1);
      if (y < h - 1) { relax(i, i + w, 1); if (x < w - 1) relax(i, i + w + 1, 1.4); if (x > 0) relax(i, i + w - 1, 1.4); }
    }
  return d;
}
