// Unit tests for web/js/markdown.js (run by tests/test_web_js.py).
import assert from "node:assert/strict";
import { renderMarkdown } from "../../web/js/markdown.js";

assert.equal(renderMarkdown("Say **I have** it."), "<p>Say <strong>I have</strong> it.</p>");
assert.equal(renderMarkdown("an *idea* here"), "<p>an <em>idea</em> here</p>");
assert.equal(renderMarkdown("Examples:\n- one\n- two\nDone."),
  "<p>Examples:</p><ul><li>one</li><li>two</li></ul><p>Done.</p>");
assert.equal(renderMarkdown("1. first\n2. second"), "<ol><li>first</li><li>second</li></ol>");
assert.equal(renderMarkdown('You said "goed" -> **went**'), "<p>You said &quot;goed&quot; → <strong>went</strong></p>");
// model output can never inject HTML
assert.equal(renderMarkdown("<img src=x onerror=alert(1)> **b**"),
  "<p>&lt;img src=x onerror=alert(1)&gt; <strong>b</strong></p>");
assert.equal(renderMarkdown("2 * 3 = 6 and 4 * 5"), "<p>2 * 3 = 6 and 4 * 5</p>");   // lone stars stay
assert.equal(renderMarkdown(""), "");
console.log("markdown tests passed");
