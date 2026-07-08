const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function postJson(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `request failed: ${res.status}`);
  return data;
}

export function postChatTrace({ message, mode, topK, topN }) {
  const body = { message };
  if (mode) body.mode = mode;
  if (topK) body.top_k = Number(topK);
  if (topN) body.top_n = Number(topN);
  return postJson("/chat/trace", body);
}

export async function getEvalResults() {
  const res = await fetch(`${BASE}/eval-results`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return res.json();
}
