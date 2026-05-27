import { useEffect, useState, useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, type GraphTopologyResponse, type TopologyNode, type TopologyEdge } from "../api/client";

// ── Status colors ──────────────────────────────────────────────────

const STATUS_BG: Record<string, string> = {
  pass: "#1a3a2a",
  fail: "#3a1a1a",
  no_evals: "#2a2a1a",
  no_artifacts: "#1a1a2a",
  no_specs: "#2a1a1a",
  unknown: "#1a1a2a",
};

const STATUS_BORDER: Record<string, string> = {
  pass: "#3fb950",
  fail: "#f85149",
  no_evals: "#d29922",
  no_artifacts: "#8b949e",
  no_specs: "#f85149",
  unknown: "#8b949e",
};

// ── Layout ─────────────────────────────────────────────────────────

function layoutNodes(rawNodes: TopologyNode[]): Node[] {
  const reqs = rawNodes.filter((n) => n.type === "requirement");
  const specs = rawNodes.filter((n) => n.type === "spec");

  const capGroups = new Map<string, TopologyNode[]>();
  for (const s of specs) {
    const cap = s.capability || s.id;
    if (!capGroups.has(cap)) capGroups.set(cap, []);
    capGroups.get(cap)!.push(s);
  }

  const nodes: Node[] = [];
  const ROW_HEIGHT = 80;

  reqs.forEach((r, i) => {
    const st = r.status || "unknown";
    nodes.push({
      id: r.id,
      position: { x: 0, y: i * ROW_HEIGHT },
      data: { label: r.label },
      style: {
        background: STATUS_BG[st] ?? STATUS_BG.unknown,
        border: `2px solid ${STATUS_BORDER[st] ?? STATUS_BORDER.unknown}`,
        borderRadius: 4,
        color: "#c9d1d9",
        fontSize: 11,
        padding: "8px 12px",
        width: 220,
      },
    });
  });

  let specRow = 0;
  for (const [, group] of capGroups) {
    for (const s of group) {
      const st = s.status || "unknown";
      const subtitle = [
        s.file_count ? `${s.file_count} files` : null,
        s.test_count ? `${s.test_count} tests` : null,
      ].filter(Boolean).join(" · ");

      nodes.push({
        id: s.id,
        position: { x: 340, y: specRow * ROW_HEIGHT },
        data: {
          label: (
            <div>
              <div>{s.label}</div>
              {subtitle && (
                <div style={{ fontSize: 9, color: "#8b949e", marginTop: 2 }}>{subtitle}</div>
              )}
            </div>
          ),
        },
        style: {
          background: STATUS_BG[st] ?? STATUS_BG.unknown,
          border: `2px solid ${STATUS_BORDER[st] ?? STATUS_BORDER.unknown}`,
          borderRadius: 8,
          color: "#c9d1d9",
          fontSize: 11,
          padding: "8px 12px",
          width: 240,
        },
      });
      specRow++;
    }
  }

  return nodes;
}

function layoutEdges(rawEdges: TopologyEdge[]): Edge[] {
  return rawEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: "smoothstep",
    animated: e.type === "DEPENDS_ON",
    style: {
      stroke: e.type === "IMPLEMENTS" ? "#3fb950" : "#58a6ff",
      strokeWidth: 1.5,
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: e.type === "IMPLEMENTS" ? "#3fb950" : "#58a6ff",
    },
    label: e.type === "DEPENDS_ON" ? "depends" : undefined,
    labelStyle: { fontSize: 9, fill: "#8b949e" },
  }));
}

// ── Component ──────────────────────────────────────────────────────

export default function DependencyGraph({ runId }: { runId: string }) {
  const [data, setData] = useState<GraphTopologyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadData = useCallback(() => {
    setLoading(true);
    api.graphTopology(runId)
      .then(setData)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => { loadData(); }, [loadData]);

  const nodes = useMemo(() => (data ? layoutNodes(data.nodes) : []), [data]);
  const edges = useMemo(() => (data ? layoutEdges(data.edges) : []), [data]);

  if (loading) return <div className="empty-state"><p>Loading dependency graph...</p></div>;
  if (error) return (
    <div className="card" style={{ borderColor: "#da3633" }}>
      <code>{error}</code>
      <button className="btn btn-secondary" onClick={loadData} style={{ marginTop: 8 }}>Retry</button>
    </div>
  );
  if (!data || data.nodes.length === 0) return <div className="empty-state"><p>No graph data for this run.</p></div>;

  const reqCount = data.nodes.filter((n) => n.type === "requirement").length;
  const specCount = data.nodes.filter((n) => n.type === "spec").length;
  const passCount = data.nodes.filter((n) => n.status === "pass").length;
  const failCount = data.nodes.filter((n) => n.status === "fail").length;

  return (
    <div>
      <div style={{ marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span style={{ color: "#8b949e", fontSize: 12 }}>
            {reqCount} requirements · {specCount} specs · {data.edges.length} edges
            {passCount > 0 && <> · <span style={{ color: "#3fb950" }}>{passCount} passing</span></>}
            {failCount > 0 && <> · <span style={{ color: "#f85149" }}>{failCount} failing</span></>}
          </span>
        </div>
        <div style={{ fontSize: 11, color: "#8b949e" }}>
          <span style={{ color: "#3fb950" }}>■</span> pass
          {" · "}<span style={{ color: "#f85149" }}>■</span> fail
          {" · "}<span style={{ color: "#d29922" }}>■</span> no evals
          {" · "}<span style={{ color: "#8b949e" }}>■</span> no artifacts
          {" · "}<span style={{ color: "#3fb950" }}>→</span> implements
          {" · "}<span style={{ color: "#58a6ff" }}>⇢</span> depends
        </div>
      </div>
      <div style={{ height: Math.max(400, nodes.length * 35 + 100), border: "1px solid #30363d", borderRadius: 8, overflow: "hidden" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          proOptions={{ hideAttribution: true }}
          style={{ background: "#0d1117" }}
        >
          <Background variant={BackgroundVariant.Dots} color="#21262d" gap={16} />
          <Controls />
          <MiniMap
            style={{ background: "#161b22" }}
            nodeColor={(n) => {
              const border = (n.style?.border as string) ?? "";
              const match = border.match(/#[0-9a-f]{6}/i);
              return match ? match[0] : "#30363d";
            }}
          />
        </ReactFlow>
      </div>
    </div>
  );
}
