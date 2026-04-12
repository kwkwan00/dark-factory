import { useCallback, useRef, useState } from "react";
import { streamAgentRun, type AGUIEvent, type AgentRunKeys } from "../api/client";

export interface Step {
  id: string;
  name: string;
  status: "pending" | "running" | "done" | "error";
  messages: string[];
}

export interface PipelineResult {
  completed_features?: Array<{
    feature: string;
    status: string;
    error?: string;
  }>;
  pass_rate?: number;
  all_artifacts?: unknown[];
  all_tests?: unknown[];
  [key: string]: unknown;
}

export type RunStatus = "idle" | "running" | "done" | "error";

export interface AgentRunState {
  status: RunStatus;
  steps: Step[];
  result: PipelineResult | null;
  error: string | null;
}

export function useAgentRun() {
  const [state, setState] = useState<AgentRunState>({
    status: "idle",
    steps: [],
    result: null,
    error: null,
  });

  const abortRef = useRef<AbortController | null>(null);

  const startRun = useCallback(
    async (requirementsPath: string, keys?: AgentRunKeys) => {
      // Abort any existing run
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      // H4 fix: pending messages tracked in local vars, not React state
      const pending: Record<string, string> = {};
      // M4 fix: track which step each message belongs to
      const msgToStep: Record<string, string> = {};
      let currentSteps: Step[] = [];

      setState({ status: "running", steps: [], result: null, error: null });

      try {
        for await (const event of streamAgentRun(requirementsPath, ctrl.signal, keys)) {
          handleEvent(event, pending, msgToStep, currentSteps, (steps) => {
            currentSteps = steps;
          });
        }
      } catch (err: unknown) {
        if ((err as { name?: string }).name === "AbortError") return;
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        }));
      }

    function handleEvent(
      event: AGUIEvent,
      pend: Record<string, string>,
      m2s: Record<string, string>,
      steps: Step[],
      setSteps: (s: Step[]) => void,
    ) {
      switch (event.type) {
        case "RUN_STARTED":
          break;

        case "RUN_FINISHED":
          setState((prev) => ({ ...prev, status: "done" }));
          break;

        case "RUN_ERROR":
          setState((prev) => ({
            ...prev,
            status: "error",
            error: event.message ?? "Unknown error",
          }));
          break;

        case "STEP_STARTED": {
          const step: Step = {
            id: event.step_id ?? crypto.randomUUID(),
            name: event.stepName ?? "Step",
            status: "running",
            messages: [],
          };
          const updated = [...steps, step];
          setSteps(updated);
          setState((prev) => ({ ...prev, steps: updated }));
          break;
        }

        case "STEP_FINISHED": {
          const updated = steps.map((s) =>
            s.id === event.step_id
              ? { ...s, status: "done" as const }
              : s
          );
          setSteps(updated);
          setState((prev) => ({ ...prev, steps: updated }));
          break;
        }

        case "TEXT_MESSAGE_START": {
          const id = event.messageId ?? "";
          pend[id] = "";
          // M4 fix: record which step was running when message started
          const runningStep = steps.findLast((s) => s.status === "running");
          if (runningStep) m2s[id] = runningStep.id;
          // No setState — intermediate bookkeeping only
          break;
        }

        case "TEXT_MESSAGE_CONTENT": {
          const id = event.messageId ?? "";
          pend[id] = (pend[id] ?? "") + (event.delta ?? "");
          // H4 fix: no setState for content deltas — avoids needless re-renders
          break;
        }

        case "TEXT_MESSAGE_END": {
          const id = event.messageId ?? "";
          const text = pend[id] ?? "";
          delete pend[id];
          // M4 fix: attach to the correct step, not just the last one
          const targetStepId = m2s[id];
          delete m2s[id];
          const updated = steps.map((s) =>
            s.id === targetStepId
              ? { ...s, messages: [...s.messages, text] }
              : s
          );
          setSteps(updated);
          setState((prev) => ({ ...prev, steps: updated }));
          break;
        }

        case "STATE_SNAPSHOT":
          setState((prev) => ({
            ...prev,
            result: event.snapshot as PipelineResult,
          }));
          break;
      }
    }
  }, []);

  // M3 fix: cancel clears all visible state
  const cancelRun = useCallback(() => {
    abortRef.current?.abort();
    setState({ status: "idle", steps: [], result: null, error: null });
  }, []);

  return { state, startRun, cancelRun };
}
