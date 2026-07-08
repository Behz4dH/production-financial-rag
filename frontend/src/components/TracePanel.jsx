const STAGE_LABELS = {
  parse_query: "1. Parse question -> entities",
  resolve_entities: "2. Resolve entities -> source filing(s)",
  retrieve_candidates: "3. Retrieve candidates",
  rerank: "4. Rerank",
  rewrite: "↻ Retry (weak relevance)",
  generate: "5. Generate answer",
  refuse: "✕ Refuse",
};

function StepBody({ stage, data }) {
  if (stage === "retrieve_candidates" || stage === "rerank") {
    const chunks = data.candidates || data.chunks || [];
    const hasScore = stage === "rerank";
    return (
      <table className="chunk-table">
        <thead>
          <tr>
            <th>source</th><th>page</th>{hasScore && <th>score</th>}<th>snippet</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((c, i) => (
            <tr key={i}>
              <td>{c.source}</td>
              <td>{c.page}</td>
              {hasScore && <td>{c.score != null ? c.score.toFixed(4) : "-"}</td>}
              <td className="snippet">{c.snippet}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return <pre className="step-data">{JSON.stringify(data, null, 2)}</pre>;
}

export default function TracePanel({ steps }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="trace-panel">
      <p className="trace-heading">Pipeline trace ({steps.length} steps)</p>
      <div className="trace-steps">
        {steps.map((s, i) => (
          <details key={i} open={i === steps.length - 1}>
            <summary>{STAGE_LABELS[s.stage] || s.stage} - {s.elapsed_ms}ms</summary>
            <StepBody stage={s.stage} data={s.data} />
          </details>
        ))}
      </div>
    </div>
  );
}
