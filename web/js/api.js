// Talking to the local Greenroom server.
// Every call goes to 127.0.0.1 (same origin), never to the internet.

export async function api(path, { method = "GET", body, form, signal } = {}) {
  const opts = { method, signal };
  if (form) {
    opts.body = form;                       // multipart: the browser sets the boundary header
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      msg = typeof data.detail === "string" ? data.detail : msg;
    } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res;
}

export const getJSON = async (path) => (await api(path)).json();
export const sendJSON = async (path, body, method = "POST") => (await api(path, { method, body })).json();

// Read an NDJSON response (one JSON object per line) and call onEvent for each
// line as soon as it arrives. This is what lets the coach start speaking
// before the whole reply exists.
export async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf));
}
