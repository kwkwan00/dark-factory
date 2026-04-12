import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import RunDetailWindow from "./components/RunDetailWindow";
import "./index.css";

// L12 fix: fail loudly if index.html is missing the #root element instead
// of silently rendering nothing via a non-null assertion.
const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error('Missing #root element in index.html — cannot mount React app');
}

// The Metrics tab opens per-run detail in a new window via
// ``window.open("/#/run-detail?run_id=...", "_blank", "popup,...")``.
// We detect that route here and render the dedicated popup component
// instead of the full App (keeps the popup lightweight and avoids
// pulling in the main tab bar / ManufactureContext).
function pickRoot(): JSX.Element {
  const hash = window.location.hash || "";
  if (hash.startsWith("#/run-detail")) {
    const qs = hash.includes("?") ? hash.split("?", 2)[1] : "";
    const params = new URLSearchParams(qs);
    const runId = params.get("run_id") ?? "";
    return <RunDetailWindow runId={runId} />;
  }
  return <App />;
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>{pickRoot()}</React.StrictMode>,
);
