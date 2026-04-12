import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "../api/client";
import { useAgentRun, type AgentRunState } from "../hooks/useAgentRun";

/** All Manufacture tab state, lifted to a context provider so it survives
 *  tab unmounts. Without this, switching tabs while a run is in progress
 *  resets the step tree, uploaded files, and path input. */
interface ManufactureContextValue {
  // Pipeline run state from useAgentRun
  state: AgentRunState;
  startRun: (requirementsPath: string) => Promise<void>;
  cancelRun: () => Promise<void>;
  cancelling: boolean;

  // Path input
  path: string;
  setPath: (p: string) => void;

  // Upload state
  uploading: boolean;
  uploadError: string | null;
  uploadedFiles: string[];
  handleFiles: (fileList: FileList | File[]) => Promise<void>;
  clearUploads: () => void;

  // Optional per-run API key overrides. Kept in in-memory React state
  // only (never localStorage) — the user must re-enter them per browser
  // session. Sent on the next run when non-empty; otherwise the server
  // uses its ANTHROPIC_API_KEY / OPENAI_API_KEY env vars.
  anthropicApiKey: string;
  setAnthropicApiKey: (key: string) => void;
  openaiApiKey: string;
  setOpenaiApiKey: (key: string) => void;
  clearApiKeys: () => void;
}

const ManufactureContext = createContext<ManufactureContextValue | null>(null);

export function ManufactureProvider({ children }: { children: ReactNode }) {
  const { state, startRun: rawStartRun, cancelRun: rawCancelRun } = useAgentRun();

  const [path, setPath] = useState("./openspec");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [anthropicApiKey, setAnthropicApiKey] = useState("");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [cancelling, setCancelling] = useState(false);
  // A ref guard for the "already cancelling" check — React state updates
  // are asynchronous, so a double-click within the same render frame can
  // pass the ``if (cancelling) return`` guard twice before the state
  // update commits. Using a ref makes the check + set synchronous.
  const cancellingRef = useRef(false);

  const handleFiles = useCallback(async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    setUploading(true);
    setUploadError(null);
    try {
      const result = await api.uploadFiles(files);
      setPath(result.path);
      setUploadedFiles(result.files);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }, []);

  const clearUploads = useCallback(() => {
    setUploadedFiles([]);
    setPath("./openspec");
    setUploadError(null);
  }, []);

  const clearApiKeys = useCallback(() => {
    setAnthropicApiKey("");
    setOpenaiApiKey("");
  }, []);

  // Wrap the raw hook's startRun so the Manufacture tab can just call
  // ``startRun(path)`` and the keys stored in context are applied
  // automatically.
  const startRun = useCallback(
    (requirementsPath: string) =>
      rawStartRun(requirementsPath, {
        anthropicApiKey: anthropicApiKey || undefined,
        openaiApiKey: openaiApiKey || undefined,
      }),
    [rawStartRun, anthropicApiKey, openaiApiKey],
  );

  // Kill-switch: server-side cancel first (so subprocesses, LLM calls, and
  // ThreadPoolExecutor workers actually stop), then client-side SSE abort.
  // The server cancel is idempotent so double-clicking is safe — but we
  // still guard here to keep the UI consistent and avoid duplicate POSTs.
  const cancelRun = useCallback(async () => {
    // Synchronous ref-based guard: React ``useState`` updates are
    // batched, so ``if (cancelling) return`` can race on a double-click
    // within 16ms. The ref provides a compare-and-set that's atomic.
    if (cancellingRef.current) return;
    cancellingRef.current = true;
    setCancelling(true);
    try {
      try {
        await api.cancelRun();
      } catch (err) {
        // If the cancel endpoint itself fails, fall through to the
        // client-side abort so the user still gets a responsive UI.
        console.warn("cancel_endpoint_failed", err);
      }
      rawCancelRun();
    } finally {
      cancellingRef.current = false;
      setCancelling(false);
    }
  }, [rawCancelRun]);

  const value: ManufactureContextValue = {
    state,
    startRun,
    cancelRun,
    cancelling,
    path,
    setPath,
    uploading,
    uploadError,
    uploadedFiles,
    handleFiles,
    clearUploads,
    anthropicApiKey,
    setAnthropicApiKey,
    openaiApiKey,
    setOpenaiApiKey,
    clearApiKeys,
  };

  return (
    <ManufactureContext.Provider value={value}>
      {children}
    </ManufactureContext.Provider>
  );
}

export function useManufacture(): ManufactureContextValue {
  const ctx = useContext(ManufactureContext);
  if (!ctx) {
    throw new Error("useManufacture must be used within a ManufactureProvider");
  }
  return ctx;
}
