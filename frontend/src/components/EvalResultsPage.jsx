import { useEffect, useState } from "react";
import { getEvalResults } from "../api.js";

export default function EvalResultsPage() {
  const [report, setReport] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getEvalResults()
      .then((r) => (r ? setReport(r) : setNotFound(true)))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (notFound) return <p className="eval-empty">No eval run yet - run `make eval` in backend/.</p>;
  if (!report) return <p>Loading...</p>;

  const modes = Object.keys(report.modes);
  const categories = [...new Set(modes.flatMap((m) => Object.keys(report.modes[m].by_category)))].sort();

  return (
    <div className="eval-page">
      <h2>Benchmark results (40-question ERC eval)</h2>
      <table className="eval-table">
        <thead>
          <tr><th>category</th>{modes.map((m) => <th key={m}>{m}</th>)}</tr>
        </thead>
        <tbody>
          {categories.map((cat) => (
            <tr key={cat}>
              <td>{cat}</td>
              {modes.map((m) => {
                const s = report.modes[m].by_category[cat];
                return (
                  <td key={m}>
                    {s ? `${Math.round(s.accuracy * 100)}% (${s.correct}/${s.total})` : "-"}
                  </td>
                );
              })}
            </tr>
          ))}
          <tr className="overall-row">
            <td>OVERALL</td>
            {modes.map((m) => <td key={m}>{Math.round(report.modes[m].overall.accuracy * 100)}%</td>)}
          </tr>
          <tr className="refusal-row">
            <td>refusal accuracy</td>
            {modes.map((m) => <td key={m}>{Math.round(report.refusal_accuracy[m] * 100)}%</td>)}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
