import { useState } from "react";
import ChatPage from "./components/ChatPage.jsx";
import EvalResultsPage from "./components/EvalResultsPage.jsx";

export default function App() {
  const [tab, setTab] = useState("chat");
  return (
    <div className="app">
      <h1>Financial-Report RAG</h1>
      <div className="tabs">
        <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
        <button className={tab === "eval" ? "active" : ""} onClick={() => setTab("eval")}>Eval Results</button>
      </div>
      {tab === "chat" ? <ChatPage /> : <EvalResultsPage />}
    </div>
  );
}
