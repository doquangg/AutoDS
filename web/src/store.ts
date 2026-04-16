import { create } from "zustand";
import { EventTypes } from "./lib/events";
import type { PipelineEvent, TargetCandidate } from "./lib/events";

type Status =
  | "idle"
  | "uploading"
  | "running"
  | "awaiting_target"
  | "complete"
  | "failed";

export interface StepCard {
  key: string; // `${node}:${pass}`
  node: string;
  pass: number;
  status: "running" | "done" | "failed";
  startedAt: string;
  durationMs?: number;
  details: PipelineEvent[];
}

export interface QAMessage {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
}

interface State {
  sessionId: string | null;
  status: Status;
  dataset: { filename: string; rows: number; cols: number } | null;
  userQuery: string | null;
  steps: StepCard[];
  currentPass: number;
  targetPrompt: {
    candidates: TargetCandidate[];
    allColumns: string[];
  } | null;
  finalAnswerTokens: string[];
  qaMessages: QAMessage[];
  error: string | null;

  // Actions
  startSession(opts: {
    sessionId: string;
    filename: string;
    rows: number;
    cols: number;
    userQuery: string;
  }): void;
  applyEvent(ev: PipelineEvent): void;
  appendQAUser(text: string): void;
  appendQAToken(text: string): void;
  finishQA(): void;
  reset(): void;
}

const initial: Omit<
  State,
  | "startSession"
  | "applyEvent"
  | "appendQAUser"
  | "appendQAToken"
  | "finishQA"
  | "reset"
> = {
  sessionId: null,
  status: "idle",
  dataset: null,
  userQuery: null,
  steps: [],
  currentPass: 0,
  targetPrompt: null,
  finalAnswerTokens: [],
  qaMessages: [],
  error: null,
};

export const useStore = create<State>((set, get) => ({
  ...initial,

  startSession({ sessionId, filename, rows, cols, userQuery }) {
    set({
      ...initial,
      sessionId,
      status: "running",
      dataset: { filename, rows, cols },
      userQuery,
    });
  },

  applyEvent(ev) {
    const s = get();
    switch (ev.type) {
      case EventTypes.NODE_STARTED: {
        const node = ev.node as string;
        const key = `${node}:${s.currentPass}`;
        set({
          steps: [
            ...s.steps,
            {
              key,
              node,
              pass: s.currentPass,
              status: "running",
              startedAt: ev.ts as string,
              details: [],
            },
          ],
        });
        return;
      }
      case EventTypes.NODE_COMPLETED: {
        const node = ev.node as string;
        const key = `${node}:${s.currentPass}`;
        set({
          steps: s.steps.map((x) =>
            x.key === key
              ? {
                  ...x,
                  status: "done",
                  durationMs: ev.duration_ms as number,
                }
              : x,
          ),
        });
        // Pass increment when sandbox completes (matches graph semantics)
        if (node === "sandbox") set({ currentPass: s.currentPass + 1 });
        return;
      }
      case EventTypes.NODE_FAILED: {
        const node = ev.node as string;
        const key = `${node}:${s.currentPass}`;
        set({
          steps: s.steps.map((x) =>
            x.key === key ? { ...x, status: "failed" } : x,
          ),
          error: (ev.error as string) ?? null,
        });
        return;
      }
      case EventTypes.TARGET_SELECTION_REQUIRED:
        set({
          status: "awaiting_target",
          targetPrompt: {
            candidates: (ev.candidates as TargetCandidate[]) ?? [],
            allColumns: (ev.all_columns as string[]) ?? [],
          },
        });
        return;
      case EventTypes.TARGET_SELECTION_RESOLVED:
        set({ status: "running", targetPrompt: null });
        return;
      case EventTypes.FINAL_ANSWER_TOKEN:
        set({
          finalAnswerTokens: [...s.finalAnswerTokens, ev.text as string],
        });
        return;
      case EventTypes.PIPELINE_COMPLETE:
        set({ status: "complete" });
        return;
      case EventTypes.PIPELINE_FAILED:
        set({
          status: "failed",
          error: (ev.error as string) ?? "Unknown error",
        });
        return;
      case EventTypes.SESSION_STARTED:
        // already handled by startSession
        return;
      default: {
        // Bind detail events to the current (last running) step
        const last = [...s.steps].reverse().find((x) => x.status === "running");
        if (last) {
          set({
            steps: s.steps.map((x) =>
              x.key === last.key
                ? { ...x, details: [...x.details, ev] }
                : x,
            ),
          });
        }
        return;
      }
    }
  },

  appendQAUser(text) {
    set((s) => ({ qaMessages: [...s.qaMessages, { role: "user", text }] }));
  },

  appendQAToken(text) {
    set((s) => {
      const last = s.qaMessages[s.qaMessages.length - 1];
      if (last && last.role === "assistant" && last.streaming) {
        return {
          qaMessages: [
            ...s.qaMessages.slice(0, -1),
            { ...last, text: last.text + text },
          ],
        };
      }
      return {
        qaMessages: [
          ...s.qaMessages,
          { role: "assistant", text, streaming: true },
        ],
      };
    });
  },

  finishQA() {
    set((s) => ({
      qaMessages: s.qaMessages.map((m, i) =>
        i === s.qaMessages.length - 1 && m.role === "assistant"
          ? { ...m, streaming: false }
          : m,
      ),
    }));
  },

  reset() {
    set({ ...initial });
  },
}));
