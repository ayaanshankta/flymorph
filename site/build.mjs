// Builds site/index.html (and page.html) from site/src.html so every code excerpt is the real, committed code.
//   <!--code:packages/core/src/otsu.ts#otsu-->   → that declaration (plus the comment right above it)
//   <!--code:package.json-->                     → the whole file
//   <!--metrics-->                               → table from the newest ml/runs/*/metrics.json
// usage: node site/build.mjs
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function excerpt(path, symbol) {
  const lines = readFileSync(join(root, path), 'utf8').split('\n');
  if (!symbol) return lines.join('\n').trimEnd();
  const decl = new RegExp(`^(export )?(async )?(function|const|class|type|def) ${symbol}\\b`);
  let start = lines.findIndex((l) => decl.test(l));
  if (start < 0) throw new Error(`${path}#${symbol} not found`);
  let end = start + 1;
  // a declaration ends at the next line that starts in column 0 and isn't a closing bracket
  while (end < lines.length && !(/^\S/.test(lines[end]) && !/^[}\])]/.test(lines[end]))) end++;
  while (end > start && (lines[end - 1].trim() === '' || /^(\/\/|#)/.test(lines[end - 1]))) end--;
  while (start > 0 && /^(\/\/|#)/.test(lines[start - 1])) start--; // keep the explanatory comment above
  return lines.slice(start, end).join('\n');
}

function metrics() {
  // newest local run (git-ignored) wins; otherwise the committed copy in ml/results
  const runs = join(root, 'ml/runs'), results = join(root, 'ml/results');
  const run = existsSync(runs) && readdirSync(runs).filter((d) => existsSync(join(runs, d, 'metrics.json'))).sort().pop();
  const saved = existsSync(results) && readdirSync(results).filter((f) => f.endsWith('metrics.json')).sort().pop();
  const file = run ? join(runs, run, 'metrics.json') : saved ? join(results, saved) : null;
  if (!file) return '<p class="note">No evaluated model yet. Run <code>ml/evaluate.py</code>.</p>';
  const latest = run || saved;
  const m = JSON.parse(readFileSync(file, 'utf8'));
  const rows = Object.entries(m.parts).map(([p, v]) =>
    `<tr><td>${p}</td><td>${v.dice.toFixed(3)}</td><td>${v.median_area_err_pct ?? '–'}%</td><td>${v.coverage_2sd ?? '–'}</td><td>${v.flies}</td></tr>`).join('');
  const warn = m.ground_truth ? '' : '<p class="warn"><b>Draft labels.</b> This model was trained and scored on SAM’s unreviewed drafts. These numbers prove the pipeline runs end to end; they are not accuracy. They will be replaced when the model is retrained on reviewed masks.</p>';
  return `${warn}<div class="scroll"><table class="num"><thead><tr><th>Part</th><th>Dice</th><th>Median area error</th><th>Inside ±2 SD</th><th>Flies</th></tr></thead><tbody>${rows}</tbody></table></div>
<p class="caption">Run <code>${latest}</code> · ${m.test_images} test images from ${m.test_sessions} sessions the model never saw · labels: ${m.labels}</p>`;
}

const src = readFileSync(join(root, 'site/src.html'), 'utf8');
const out = src
  .replace(/<!--code:([^#>]+?)(?:#(\w+))?-->/g, (_, p, s) =>
    `<figure class="code"><figcaption>${p}${s ? ` · ${s}` : ''}</figcaption><pre><code class="language-${p.split('.').pop().replace('tsx', 'typescript').replace(/^ts$/, 'typescript').replace(/^py$/, 'python').replace('mjs', 'javascript')}">${esc(excerpt(p, s))}</code></pre></figure>`)
  .replace('<!--metrics-->', metrics);
// index.html is a full document (open it locally or host on GitHub Pages); page.html is the bare page
// content, for hosts that add their own <!doctype>/<head> wrapper.
writeFileSync(join(root, 'site/index.html'), `<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">\n${out}\n</html>\n`);
writeFileSync(join(root, 'site/page.html'), out);
console.log('site/index.html + site/page.html written');
