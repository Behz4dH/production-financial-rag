import { useState } from "react";
import { postChatTrace } from "../api.js";
import TracePanel from "./TracePanel.jsx";

export default function ChatPage() {
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState("");
  const [topK, setTopK] = useState("");
  const [topN, setTopN] = useState("");
  const [history, setHistory] = useState([]); // newest first: [{id, question, response}]
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk(e) {
    e.preventDefault();
    if (!message.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await postChatTrace({ message, mode, topK, topN });
      setHistory((h) => [{ id: crypto.randomUUID(), question: message, response }, ...h]);
      setMessage("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-page">
      <form onSubmit={handleAsk} className="chat-form">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='e.g. What was the net income of "Petra Diamonds" in fiscal year 2022?'
        />
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="">default mode</option>
          <option value="basic">basic</option>
          <option value="hybrid">hybrid</option>
        </select>
        <input value={topK} onChange={(e) => setTopK(e.target.value)} placeholder="top_k" className="num" />
        <input value={topN} onChange={(e) => setTopN(e.target.value)} placeholder="top_n" className="num" />
        <button type="submit" disabled={loading}>{loading ? "Asking..." : "Ask"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      <div className="history">
        {history.map((h) => (
          <div key={h.id} className="qa-block">
            <p className="question">Q: {h.question}</p>
            <p className={"answer" + (h.response.refused ? " refused" : "")}>
              A: {h.response.response}
              {h.response.refused && h.response.refusal_reason && (
                <span className="reason"> ({h.response.refusal_reason})</span>
              )}
            </p>
            {h.response.citations && h.response.citations.length > 0 && (
              <div className="citations">
                {h.response.citations.map((c, j) => (
                  <span key={j} className="chip">{c.source} p{c.page}</span>
                ))}
              </div>
            )}
            <TracePanel steps={h.response.steps} />
          </div>
        ))}
      </div>
    </div>
  );
}
