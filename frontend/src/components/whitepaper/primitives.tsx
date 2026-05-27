import { type ReactNode } from "react";
import {
  Background,
  BackgroundVariant,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// ─────────────────────────────────────────────────────────────────────────────
// Whitepaper primitives — shared by every architectural whitepaper rendered
// inside the Dark Factory dashboard (Dark Factory itself, the Requirements
// Refinery, future ones). These were extracted from ``AboutTab.tsx`` so all
// whitepapers share one visual language.
//
// Two layers:
//
// 1. **Section primitives** — ``Section`` / ``SubHeading`` / ``Para`` /
//    ``Callout`` / ``FactTable`` / ``Code``. Plain typography + layout.
// 2. **Diagram primitives** — a small ReactFlow wrapper (``FlowFigure``)
//    plus helpers (``makeNode`` / ``makeEdge`` / ``Label``) and a category
//    palette (``CATEGORY_STYLES``) so every diagram in every whitepaper uses
//    the same shape vocabulary (UI / API / service / phase / memory / agent
//    / terminal).
//
// Authoring a new whitepaper means importing from this module and writing
// pure content; no styling decisions need to be made per-page.
// ─────────────────────────────────────────────────────────────────────────────

// ── Section primitives ──────────────────────────────────────────────────────

interface SectionProps {
  id?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export function Section({ id, title, subtitle, children }: SectionProps) {
  return (
    <div className="card" id={id} style={{ marginBottom: 16 }}>
      <div className="card-title">{title}</div>
      {subtitle && (
        <p
          style={{
            opacity: 0.6,
            fontSize: 13,
            margin: "0 0 12px",
            fontStyle: "italic",
          }}
        >
          {subtitle}
        </p>
      )}
      {children}
    </div>
  );
}

interface CalloutProps {
  tone?: "info" | "warn" | "success";
  title?: string;
  children: ReactNode;
}

export function Callout({ tone = "info", title, children }: CalloutProps) {
  const border =
    tone === "warn" ? "#9a6700" : tone === "success" ? "#1a7f37" : "#0550ae";
  return (
    <div
      className="callout"
      style={{
        margin: "12px 0",
        padding: 12,
        borderLeft: `3px solid ${border}`,
        background: "#f0f4f8",
        borderRadius: 4,
        fontSize: 13,
      }}
    >
      {title && (
        <div
          style={{
            fontWeight: 600,
            color: border,
            marginBottom: 4,
            fontSize: 12,
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function Para({ children }: { children: ReactNode }) {
  return (
    <p
      style={{
        margin: "0 0 10px",
        fontSize: 14,
        lineHeight: 1.65,
      }}
    >
      {children}
    </p>
  );
}

export function SubHeading({ children }: { children: ReactNode }) {
  return (
    <h3
      style={{
        fontSize: 14,
        color: "#0550ae",
        margin: "16px 0 6px",
        textTransform: "uppercase",
        letterSpacing: 0.6,
        fontWeight: 600,
      }}
    >
      {children}
    </h3>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code>{children}</code>;
}

interface FactTableProps {
  rows: Array<[string, string]>;
}

export function FactTable({ rows }: FactTableProps) {
  return (
    <table className="table"
      style={{
        fontSize: 13,
        margin: "8px 0 12px",
      }}
    >
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td
              style={{
                padding: "6px 12px 6px 0",
                opacity: 0.6,
                width: "35%",
                verticalAlign: "top",
              }}
            >
              {k}
            </td>
            <td style={{ padding: "6px 0" }}>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── ReactFlow helpers ───────────────────────────────────────────────────────

export type Category =
  | "ui"
  | "api"
  | "service"
  | "phase"
  | "memory"
  | "agent"
  | "terminal";

export const CATEGORY_STYLES: Record<
  Category,
  { bg: string; border: string; color: string }
> = {
  ui: { bg: "#dbeafe", border: "#2563eb", color: "#1e40af" },
  api: { bg: "#dcfce7", border: "#16a34a", color: "#166534" },
  service: { bg: "#fef3c7", border: "#d97706", color: "#92400e" },
  phase: { bg: "#ede9fe", border: "#7c3aed", color: "#5b21b6" },
  memory: { bg: "#fce7f3", border: "#db2777", color: "#9d174d" },
  agent: { bg: "#ffedd5", border: "#ea580c", color: "#9a3412" },
  terminal: { bg: "#f3f4f6", border: "#9ca3af", color: "#4b5563" },
};

interface NodeOpts {
  width?: number;
  height?: number;
  horizontal?: boolean;
}

export function makeNode(
  id: string,
  label: ReactNode,
  x: number,
  y: number,
  cat: Category,
  opts: NodeOpts = {},
): Node {
  const { width = 160, height, horizontal = false } = opts;
  const s = CATEGORY_STYLES[cat];
  return {
    id,
    position: { x, y },
    data: { label },
    sourcePosition: horizontal ? Position.Right : Position.Bottom,
    targetPosition: horizontal ? Position.Left : Position.Top,
    style: {
      background: s.bg,
      border: `1.5px solid ${s.border}`,
      color: s.color,
      borderRadius: 6,
      fontSize: 11,
      fontFamily:
        "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
      padding: "8px 10px",
      width,
      ...(height ? { height } : {}),
      textAlign: "center" as const,
      lineHeight: 1.35,
    },
    draggable: false,
    selectable: false,
    connectable: false,
  };
}

interface EdgeOpts {
  label?: string;
  dashed?: boolean;
  color?: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export function makeEdge(
  id: string,
  source: string,
  target: string,
  opts: EdgeOpts = {},
): Edge {
  const stroke = opts.color ?? "#9ca3af";
  return {
    id,
    source,
    target,
    type: "smoothstep",
    animated: false,
    label: opts.label,
    labelStyle: {
      fill: "#4b5563",
      fontSize: 10,
      fontFamily:
        "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    },
    labelBgStyle: { fill: "#f8f9fa" },
    labelBgPadding: [4, 2],
    labelBgBorderRadius: 2,
    style: {
      stroke,
      strokeWidth: 1.5,
      ...(opts.dashed ? { strokeDasharray: "5 4" } : {}),
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
  };
}

interface FlowFigureProps {
  nodes: Node[];
  edges: Edge[];
  caption?: string;
  height?: number;
}

export function FlowFigure({
  nodes,
  edges,
  caption,
  height = 340,
}: FlowFigureProps) {
  return (
    <figure
      style={{
        margin: "16px 0",
        padding: 0,
        background: "#f8f9fa",
        border: "1px solid #d0d7de",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      <div style={{ height, width: "100%" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
          minZoom={0.1}
          maxZoom={2}
          style={{ background: "transparent" }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            color="#d0d7de"
            gap={18}
            size={1}
          />
        </ReactFlow>
      </div>
      {caption && (
        <figcaption
          style={{
            padding: "10px 16px 12px",
            color: "#57606a",
            fontSize: 12,
            fontStyle: "italic",
            textAlign: "center",
            borderTop: "1px solid #d0d7de",
            background: "#f0f4f8",
          }}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

// Multi-line node label — bolds the first line, dims subsequent lines.
export function Label({ lines }: { lines: string[] }) {
  return (
    <div>
      {lines.map((l, i) => (
        <div
          key={i}
          style={{
            fontWeight: i === 0 ? 600 : 400,
            opacity: i === 0 ? 1 : 0.75,
            fontSize: i === 0 ? 11 : 10,
          }}
        >
          {l}
        </div>
      ))}
    </div>
  );
}
