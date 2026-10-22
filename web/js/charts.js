// Tiny inline-SVG line chart for the Progress screen.
// Hand-written instead of a chart library because the app must work offline
// (no CDN) and only needs one simple chart type.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// points: [{label, value}] in time order (value may be null = no data that session)
export function lineChart(points, { decimals = 2, unit = "" } = {}) {
  const data = points.filter((p) => p.value !== null && p.value !== undefined);
  if (!data.length) return `<p class="muted small">No data yet.</p>`;
  const W = 320, H = 150, L = 34, R = 10, T = 12, B = 26;
  const vals = data.map((p) => p.value);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo = Math.max(0, lo - 1); hi = hi + 1; }   // flat line: give it room
  if (lo > 0 && lo < hi * 0.5) lo = 0;   // start the axis at 0 when that is close anyway
  const x = (i) => L + (data.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (data.length - 1));
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const path = data.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const fmt = (v) => `${Number(v).toFixed(decimals)}${unit}`;
  const ticks = [lo, (lo + hi) / 2, hi];
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="trend over ${data.length} sessions">
    ${ticks.map((t) => `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/>
      <text x="${L - 4}" y="${y(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
    <path class="line" d="${path}"/>
    ${data.map((p, i) => `<circle class="pt" cx="${x(i)}" cy="${y(p.value)}" r="3.5"><title>${esc(p.label)}: ${fmt(p.value)}</title></circle>`).join("")}
    <text x="${L}" y="${H - 6}">${esc(data[0].label)}</text>
    ${data.length > 1 ? `<text x="${W - R}" y="${H - 6}" text-anchor="end">${esc(data[data.length - 1].label)}</text>` : ""}
  </svg>`;
}
