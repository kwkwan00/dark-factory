import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";

type FetchState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; data: T }
  | { status: "error"; error: string };

/** Default request timeout (30 s). Prevents indefinite hangs when the
 * backend is slow or unresponsive. */
const DEFAULT_TIMEOUT_MS = 30_000;

// M1 fix: use ref for fetcher so `load` always calls the latest closure.
// M8 fix: guard against setState on unmounted component.
function useFetch<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  timeoutMs = DEFAULT_TIMEOUT_MS,
) {
  const [state, setState] = useState<FetchState<T>>({ status: "idle" });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const mountedRef = useRef(true);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => { mountedRef.current = false; }, []);

  const load = useCallback(async () => {
    // Abort any in-flight request before starting a new one.
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);

    setState({ status: "loading" });
    try {
      const data = await fetcherRef.current();
      if (mountedRef.current && !ctrl.signal.aborted) {
        setState({ status: "done", data });
      }
    } catch (e) {
      if (ctrl.signal.aborted) return; // timed out or superseded — ignore
      if (mountedRef.current) {
        setState({
          status: "error",
          error: e instanceof Error ? e.message : String(e),
        });
      }
    } finally {
      clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { state, load };
}

export function useGraphGaps(staleDays = 7) {
  const { state, load } = useFetch(() => api.graphGaps(staleDays), [staleDays]);
  useEffect(() => {
    void load();
  }, [load]);
  return { state, refresh: load };
}

export function useHistory(limit = 20) {
  return useFetch(() => api.history(limit), [limit]);
}

export function useAgentMemorySearch() {
  const [keywords, setKeywords] = useState("");
  const [type, setType] = useState("all");
  const { state, load } = useFetch(
    () => api.memorySearch(keywords, type),
    [keywords, type],
  );
  return { state, keywords, setKeywords, type, setType, search: load };
}

export function useAgentMemoryList(initialType = "all", initialLimit = 100) {
  const [type, setType] = useState(initialType);
  const [limit, setLimit] = useState(initialLimit);
  const { state, load } = useFetch(
    () => api.memoryList(type, limit),
    [type, limit],
  );

  // Auto-load on mount and whenever type/limit change
  useEffect(() => {
    void load();
  }, [load]);

  return { state, type, setType, limit, setLimit, refresh: load };
}

/** Fetch the evaluations for a single pipeline run. Used by the
 * Run Detail popup's Evaluations screen — replaces the deleted
 * top-level Evaluations tab. */
export function useRunEvaluation(runId: string) {
  const { state, load } = useFetch(
    () => api.evalRuns(20, runId),
    [runId],
  );
  useEffect(() => {
    if (runId) void load();
  }, [runId, load]);
  return { state, refresh: load };
}

export function useHealth() {
  const { state, load } = useFetch(() => api.health(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function usePipelineSettings() {
  const { state, load } = useFetch(() => api.getSettings(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}


export function useMetricsSummary() {
  const { state, load } = useFetch(() => api.metricsSummary(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsRuns(limit = 20) {
  const { state, load } = useFetch(() => api.metricsRuns(limit), [limit]);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsEvalTrend(metricName?: string, limit = 200) {
  const { state, load } = useFetch(
    () => api.metricsEvalTrend(metricName, limit),
    [metricName, limit],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsLlmUsage(
  groupBy: "model" | "phase" | "client" = "model",
) {
  const { state, load } = useFetch(() => api.metricsLlmUsage(groupBy), [groupBy]);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsSwarmFeatures(runId?: string, limit = 100) {
  const { state, load } = useFetch(
    () => api.metricsSwarmFeatures(runId, limit),
    [runId, limit],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsCostRollup() {
  const { state, load } = useFetch(() => api.metricsCostRollup(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsThroughput(days = 30) {
  const { state, load } = useFetch(() => api.metricsThroughput(days), [days]);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsQuality() {
  const { state, load } = useFetch(() => api.metricsQuality(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsIncidents(unresolvedOnly = false) {
  const { state, load } = useFetch(
    () => api.metricsIncidents({ unresolvedOnly }),
    [unresolvedOnly],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsAgentStats(runId?: string) {
  const { state, load } = useFetch(
    () => api.metricsAgentStats(runId),
    [runId],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsToolCalls(
  groupBy: "tool" | "agent" | "feature" = "tool",
) {
  const { state, load } = useFetch(
    () => api.metricsToolCalls(groupBy),
    [groupBy],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsMemoryActivity() {
  const { state, load } = useFetch(() => api.metricsMemoryActivity(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

/** Tier A: procedural memory graph observability. Counts by type,
 * relevance histogram, top-10 recalled, recall effectiveness. */
export function useMetricsMemory() {
  const { state, load } = useFetch(() => api.memoryMetrics(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsDecomposition() {
  const { state, load } = useFetch(() => api.metricsDecomposition(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsArtifacts() {
  const { state, load } = useFetch(() => api.metricsArtifacts(), []);
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}

export function useMetricsBackgroundLoop(limit = 200) {
  const { state, load } = useFetch(
    () => api.metricsBackgroundLoop(limit),
    [limit],
  );
  useEffect(() => { void load(); }, [load]);
  return { state, refresh: load };
}


export function useWatcher() {
  const { state: statusState, load: loadStatus } = useFetch(
    () => api.watchStatus(),
    [],
  );

  useEffect(() => { void loadStatus(); }, [loadStatus]);

  const start = useCallback(async () => {
    await api.watchStart();
    await loadStatus();
  }, [loadStatus]);

  const stop = useCallback(async () => {
    await api.watchStop();
    await loadStatus();
  }, [loadStatus]);

  return { statusState, start, stop, refresh: loadStatus };
}
