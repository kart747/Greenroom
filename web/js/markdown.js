// A tiny, safe markdown renderer for the tutor's replies.
// Supports only what the tutor is told to use (personas.py): **bold**,
// *italic*, `code`, "- " and "1. " lists, and paragraphs.
// Safety: the text is HTML-escaped FIRST, then the few markdown patterns are
// turned into tags, so nothing the model writes can inject HTML or scripts.
// Hand-written instead of a library because the app is offline (no CDN) and
// this is all it needs.

const escapeHTML = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/-&gt;/g, "→");
}

export function renderMarkdown(src) {
  let html = "";
  let list = null;                       // "ul" or "ol" while inside a list
  for (const raw of escapeHTML(src).split("\n")) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const kind = ul ? "ul" : ol ? "ol" : null;
    if (list && kind !== list) { html += `</${list}>`; list = null; }
    if (kind) {
      if (!list) { html += `<${kind}>`; list = kind; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (line.trim()) {
      html += `<p>${inline(line.trim())}</p>`;
    }
  }
  if (list) html += `</${list}>`;
  return html;
}
