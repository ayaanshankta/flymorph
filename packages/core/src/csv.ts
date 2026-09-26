type Row = Record<string, string | number | undefined>;

// RFC 4180: wrap a cell in quotes if it contains a comma, quote or newline; double any quotes inside.
const cell = (v: unknown) => {
  const s = v === undefined ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function toCSV(rows: Row[]): string {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  return [cols.join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))].join('\n') + '\n';
}
