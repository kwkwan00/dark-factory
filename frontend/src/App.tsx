import { Component, type ReactNode, useState } from "react";
import AboutTab from "./components/AboutTab";
import AgentLogsTab from "./components/AgentLogsTab";
import AgentMemoryTab from "./components/AgentMemoryTab";
import GapFinderTab from "./components/GapFinderTab";
import ManufactureTab from "./components/ManufactureTab";
import MetricsTab from "./components/MetricsTab";
import SettingsTab from "./components/SettingsTab";
import { ManufactureProvider } from "./contexts/ManufactureContext";

// ── L4 fix: Error Boundary ───────────────────────────────────────────────────

interface EBProps { children: ReactNode }
interface EBState { error: Error | null }

class ErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { error: null };

  static getDerivedStateFromError(error: Error): EBState {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card" style={{ margin: 24, borderColor: "#da3633" }}>
          <div className="card-title" style={{ color: "#f85149" }}>
            Something went wrong
          </div>
          <code style={{ display: "block", marginBottom: 12 }}>
            {this.state.error.message}
          </code>
          <button
            className="btn btn-secondary"
            onClick={() => this.setState({ error: null })}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── App ───────────────────────────────────────────────────────────────────────

type Tab =
  | "manufacture"
  | "agent-logs"
  | "gap-finder"
  | "agent-memory"
  | "metrics"
  | "settings"
  | "about";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "manufacture", label: "Manufacture" },
  { id: "agent-logs", label: "Agent Logs" },
  { id: "gap-finder", label: "Gap Finder" },
  { id: "agent-memory", label: "Agent Memory" },
  { id: "metrics", label: "Metrics" },
  { id: "settings", label: "Settings" },
  { id: "about", label: "About" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("manufacture");

  return (
    <ManufactureProvider>
      <header className="app-header">
        <h1>AI Dark Factory</h1>
        <span className="badge">L4 Agent</span>
      </header>

      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn${activeTab === tab.id ? " active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="tab-content">
        <ErrorBoundary>
          {activeTab === "manufacture" && <ManufactureTab />}
          {activeTab === "agent-logs" && <AgentLogsTab />}
          {activeTab === "gap-finder" && <GapFinderTab />}
          {activeTab === "agent-memory" && <AgentMemoryTab />}
          {activeTab === "metrics" && <MetricsTab />}
          {activeTab === "settings" && <SettingsTab />}
          {activeTab === "about" && <AboutTab />}
        </ErrorBoundary>
      </main>
    </ManufactureProvider>
  );
}
