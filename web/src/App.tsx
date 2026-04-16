import { useEffect, useRef } from "react";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { RefreshIcon, SparklesIcon } from "./components/Icons";
import { useStore } from "./store";
import { askQuestion, createSession } from "./lib/api";
import { subscribe } from "./lib/sse";

export default function App() {
  const sessionId = useStore((s) => s.sessionId);
  const status = useStore((s) => s.status);
  const startSession = useStore((s) => s.startSession);
  const applyEvent = useStore((s) => s.applyEvent);
  const appendQAUser = useStore((s) => s.appendQAUser);
  const appendQAToken = useStore((s) => s.appendQAToken);
  const finishQA = useStore((s) => s.finishQA);
  const reset = useStore((s) => s.reset);

  useEffect(() => {
    if (!sessionId) return;
    const close = subscribe(sessionId, applyEvent);
    return close;
  }, [sessionId, applyEvent]);

  // Auto-scroll to bottom when new messages arrive
  const scrollRef = useRef<HTMLDivElement>(null);
  const steps = useStore((s) => s.steps);
  const qaCount = useStore((s) => s.qaMessages.length);
  const finalLen = useStore((s) => s.finalAnswerTokens.length);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [steps.length, qaCount, finalLen, status]);

  async function handleInitial(file: File, query: string) {
    try {
      const r = await createSession(file, query);
      startSession({
        sessionId: r.session_id,
        filename: r.filename,
        rows: r.rows,
        cols: r.cols,
        userQuery: query,
      });
    } catch (e) {
      console.error(e);
      alert(`Failed to start session: ${e instanceof Error ? e.message : e}`);
    }
  }

  async function handleFollowup(q: string) {
    if (!sessionId) return;
    appendQAUser(q);
    const resp = await askQuestion(sessionId, q);
    if (!resp.ok || !resp.body) return;
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const m = /^data: (.+)$/m.exec(line);
        if (!m) continue;
        try {
          const ev = JSON.parse(m[1]);
          if (ev.type === "qa_token") appendQAToken(ev.text);
          else if (ev.type === "qa_complete") finishQA();
        } catch {
          // skip malformed
        }
      }
    }
  }

  const composerDisabled =
    status === "running" || status === "awaiting_target" || status === "uploading";

  return (
    <div className="h-full flex flex-col">
      <header className="sticky top-0 z-10 bg-canvas/90 backdrop-blur-md
        border-b border-border">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center
          justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent
                to-accentHover text-white flex items-center justify-center
                shadow-sm"
            >
              <SparklesIcon size={14} />
            </div>
            <div className="font-semibold tracking-tight text-ink">
              AutoDS
            </div>
            <StatusPill status={status} />
          </div>
          {sessionId && (
            <button
              onClick={reset}
              className="flex items-center gap-1.5 text-[13px] text-muted
                hover:text-ink px-2.5 py-1 rounded-md hover:bg-border/40
                transition"
            >
              <RefreshIcon size={13} />
              New session
            </button>
          )}
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <ChatThread />
      </div>

      <div className="sticky bottom-0 bg-gradient-to-t from-canvas via-canvas
        to-canvas/0 pt-6 pb-4">
        <div className="max-w-3xl mx-auto px-4">
          <Composer
            mode={status === "complete" ? "followup" : "initial"}
            disabled={composerDisabled}
            onSubmitInitial={handleInitial}
            onSubmitFollowup={handleFollowup}
          />
          <div className="text-center text-[11px] text-subtle mt-2">
            AutoDS can make mistakes. Verify important decisions against the
            pipeline artifacts.
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  if (status === "idle") return null;
  const styles: Record<string, string> = {
    running: "bg-running/10 text-running",
    awaiting_target: "bg-accent/10 text-accent",
    complete: "bg-success/10 text-success",
    failed: "bg-danger/10 text-danger",
    uploading: "bg-muted/10 text-muted",
  };
  const labels: Record<string, string> = {
    running: "Running",
    awaiting_target: "Your input needed",
    complete: "Complete",
    failed: "Failed",
    uploading: "Uploading",
  };
  const label = labels[status];
  if (!label) return null;
  return (
    <span
      className={`ml-1 text-[11px] font-medium px-2 py-0.5 rounded-full
        ${styles[status] ?? "bg-canvas text-muted"}`}
    >
      {label}
    </span>
  );
}
