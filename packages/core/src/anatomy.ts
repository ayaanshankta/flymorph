import { components } from './ccl';
import { gray } from './image';

// A fly's thorax sits BETWEEN its head and abdomen, so seen from the thorax's centre the head and the
// abdomen point in opposite directions: the angle between them is over 90°, i.e. their dot product is
// negative. A label map where both point the same way is almost certainly wrong (e.g. a head painted as
// abdomen). Returns the thorax centre of every fly that breaks the rule; flies missing a part are skipped.
export function misorderedFlies(labels: ArrayLike<number>, w: number, h: number): { x: number; y: number }[] {
  const fly = gray(w, h);
  for (let i = 0; i < w * h; i++) fly.data[i] = labels[i] ? 1 : 0;
  const { labels: inst, comps } = components(fly);
  const bad: { x: number; y: number }[] = [];
  for (const c of comps) {
    const sum = [[0, 0], [0, 0], [0, 0], [0, 0]], n = [0, 0, 0, 0]; // per label: Σx, Σy, count
    for (let y = c.y; y < c.y + c.h; y++)
      for (let x = c.x; x < c.x + c.w; x++) {
        const i = y * w + x, k = labels[i];
        if (inst[i] !== c.id || !k) continue;
        sum[k][0] += x; sum[k][1] += y; n[k]++;
      }
    if (!n[1] || !n[2] || !n[3]) continue;
    const [hx, hy] = [sum[1][0] / n[1], sum[1][1] / n[1]];
    const [tx, ty] = [sum[2][0] / n[2], sum[2][1] / n[2]];
    const [ax, ay] = [sum[3][0] / n[3], sum[3][1] / n[3]];
    if ((hx - tx) * (ax - tx) + (hy - ty) * (ay - ty) >= 0) bad.push({ x: tx, y: ty });
  }
  return bad;
}
