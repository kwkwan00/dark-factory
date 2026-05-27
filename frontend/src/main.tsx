import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import AboutDarkFactory from "./components/AboutDarkFactory";
import AboutRequirementsRefinery from "./components/AboutRequirementsRefinery";
import RunCompareWindow from "./components/RunCompareWindow";
import RunDetailWindow from "./components/RunDetailWindow";
import "./index.css";

// L12 fix: fail loudly if index.html is missing the #root element instead
// of silently rendering nothing via a non-null assertion.
const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error('Missing #root element in index.html — cannot mount React app');
}

// Popup windows use hash routes so the same Vite bundle can render
// lightweight popup components instead of the full App.
function pickRoot(): JSX.Element {
  const hash = window.location.hash || "";
  if (hash.startsWith("#/run-detail")) {
    const qs = hash.includes("?") ? hash.split("?", 2)[1] : "";
    const params = new URLSearchParams(qs);
    const runId = params.get("run_id") ?? "";
    return <RunDetailWindow runId={runId} />;
  }
  if (hash.startsWith("#/run-compare")) {
    const qs = hash.includes("?") ? hash.split("?", 2)[1] : "";
    const params = new URLSearchParams(qs);
    const runA = params.get("run_a") ?? "";
    const runB = params.get("run_b") ?? "";
    return <RunCompareWindow runA={runA} runB={runB} />;
  }
  if (hash.startsWith("#/about-dark-factory")) {
    return <AboutDarkFactory />;
  }
  if (hash.startsWith("#/about-requirements-refinery")) {
    return <AboutRequirementsRefinery />;
  }
  return <App />;
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>{pickRoot()}</React.StrictMode>,
);
