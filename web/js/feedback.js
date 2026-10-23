// Showing analyzer findings inside the learner's own sentence.
// Used by the report screen and, in Tutor mode, under each chat message.
//
// highlight() cuts the sentence at every finding boundary, so overlapping
// findings (a vocabulary span inside a grammar span) each stay visible.
// A highlighted part opens a popover with the message and suggestions on
// hover (mouse), tap (touch) or keyboard focus.

export const ANALYZER_LABEL = { vocabulary: "Word choice", fluency: "Fluency", grammar: "Grammar", pace: "Pace" };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function highlight(text, findings) {
  const cuts = new Set([0, text.length]);
  findings.forEach((f) => { cuts.add(f.start); cuts.add(f.end); });
  const pts = [...cuts].sort((a, b) => a - b);
  let html = "";
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const idx = findings.map((f, j) => (f.start <= a && f.end >= b ? j : -1)).filter((j) => j >= 0);
    const piece = esc(text.slice(a, b));
    html += idx.length
      ? `<mark class="${findings[idx[0]].analyzer}" tabindex="0" data-f="${idx.join(",")}">${piece}</mark>`
      : piece;
  }
  return html;
}

export function findingsHTML(findings) {
  return findings.map((f) => `
    <div><b>${esc(ANALYZER_LABEL[f.analyzer] || f.analyzer)}:</b> ${esc(f.message)}
      ${f.suggestions?.length ? `<div class="try">Try: ${f.suggestions.map((s) => `<b>${esc(s)}</b>`).join(", ")}</div>` : ""}
    </div>`).join("<hr>");
}

// One popover at a time; any click elsewhere or a screen change closes it.
let pop = null;
export function closePopover() {
  pop?.remove();
  pop = null;
  document.querySelectorAll("mark.open").forEach((m) => m.classList.remove("open"));
}
document.addEventListener("click", closePopover);
window.addEventListener("hashchange", closePopover);

function openPopover(mark) {
  closePopover();
  // The element holding the sentence keeps its findings as a JS property.
  const host = mark.closest("[data-findings]");
  const fs = mark.dataset.f.split(",").map((j) => host._findings[Number(j)]);
  pop = document.createElement("div");
  pop.className = "popover";
  pop.innerHTML = findingsHTML(fs);
  document.body.appendChild(pop);
  const r = mark.getBoundingClientRect();
  const maxLeft = document.documentElement.clientWidth - pop.offsetWidth - 8;
  pop.style.left = `${window.scrollX + Math.max(8, Math.min(r.left, maxLeft))}px`;
  pop.style.top = `${window.scrollY + r.bottom + 6}px`;
  mark.classList.add("open");
}

// Render `text` with its findings into `host` and make the marks interactive.
export function showFindings(host, text, findings) {
  host.dataset.findings = "";
  host._findings = findings;
  host.innerHTML = highlight(text, findings);
  if (host._wired) return;
  host._wired = true;
  host.addEventListener("mouseover", (e) => {
    const m = e.target.closest("mark");
    if (m && !m.classList.contains("open")) openPopover(m);
  });
  host.addEventListener("click", (e) => {
    const m = e.target.closest("mark");
    if (!m) return;
    e.stopPropagation();   // keep the document handler from closing it right away
    if (m.classList.contains("open") && e.pointerType !== "mouse") closePopover();
    else openPopover(m);
  });
  host.addEventListener("focusin", (e) => {
    const m = e.target.closest("mark");
    if (m) openPopover(m);
  });
}
