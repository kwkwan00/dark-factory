import RunCompare from "./RunCompare";

interface RunCompareWindowProps {
  runA: string;
  runB: string;
}

export default function RunCompareWindow({ runA, runB }: RunCompareWindowProps) {
  if (!runA || !runB) {
    return (
      <div style={{ padding: 24, fontFamily: "system-ui", color: "#c9d1d9", background: "#0d1117", minHeight: "100vh" }}>
        <h2 style={{ color: "#f85149" }}>Missing run IDs</h2>
        <p>
          This window requires two run IDs. Open it via the Compare button in
          the Manufacture tab.
          <br />
          Example: <code>#/run-compare?run_a=run-123&run_b=run-456</code>
        </p>
      </div>
    );
  }

  return (
    <div style={{ padding: 16, fontFamily: "system-ui", color: "#c9d1d9", background: "#0d1117", minHeight: "100vh" }}>
      <RunCompare
        runA={runA}
        runB={runB}
        onClose={() => window.close()}
      />
    </div>
  );
}
