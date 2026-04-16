import { useEffect, useRef } from "react";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { LogoMark, PlusIcon } from "./components/Icons";
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
  const failQA = useStore((s) => s.failQA);
  const reset = useStore((s) => s.reset);

  useEffect(() => {
    if (!sessionId) return;
    const close = subscribe(sessionId, applyEvent);
    return close;
  }, [sessionId, applyEvent]);

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

    let resp: Response;
    try {
      resp = await askQuestion(sessionId, q);
    } catch (e) {
      failQA(`Request failed: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      failQA(
        `Server returned ${resp.status}${body ? `: ${body.slice(0, 240)}` : ""}`,
      );
      return;
    }
    if (!resp.body) {
      failQA("Server returned no response body");
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let sawAnything = false;
    let sawComplete = false;
    try {
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
            sawAnything = true;
            if (ev.type === "qa_token") appendQAToken(ev.text);
            else if (ev.type === "qa_complete") {
              finishQA();
              sawComplete = true;
            } else if (ev.type === "qa_error") {
              failQA(
                typeof ev.error === "string"
                  ? ev.error
                  : "The Q&A agent reported an error",
              );
              sawComplete = true;
            }
          } catch {
            // skip malformed
          }
        }
      }
    } catch (e) {
      failQA(
        `Stream failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      return;
    }

    // Stream ended without any qa_complete/qa_error — surface something
    // instead of leaving a blinking cursor with no content.
    if (!sawComplete) {
      failQA(
        sawAnything
          ? "The response ended unexpectedly before completing."
          : "The server closed the connection without sending any response. Check the backend logs for Q&A errors.",
      );
    }
  }

  const composerDisabled =
    status === "running" || status === "awaiting_target" || status === "uploading";
  const hasConversation = !!sessionId;

  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="sticky top-0 z-20 bg-canvas/80 backdrop-blur-md">
        <div className="max-w-3xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <LogoMark size={26} />
            <div className="font-semibold tracking-tight text-ink text-[15px]">
              AutoDS
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusPill status={status} />
            {hasConversation && (
              <button
                onClick={reset}
                className="flex items-center gap-1.5 text-[13px] font-medium
                  text-inkSoft hover:text-ink px-3 py-1.5 rounded-full
                  border border-border hover:border-borderStrong bg-surface
                  hover:bg-sidebar transition"
              >
                <PlusIcon size={13} />
                New chat
              </button>
            )}
          </div>
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <ChatThread />
      </div>

      <div className="relative pt-4 pb-5">
        <div className="pointer-events-none absolute inset-x-0 -top-6 h-6
          bg-gradient-to-t from-canvas to-transparent" />
        <div className="max-w-3xl mx-auto px-4">
          <Composer
            mode={status === "complete" ? "followup" : "initial"}
            disabled={composerDisabled}
            onSubmitInitial={handleInitial}
            onSubmitFollowup={handleFollowup}
          />
          <div className="text-center text-[11px] text-subtle mt-2.5">
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
    awaiting_target: "Needs input",
    complete: "Complete",
    failed: "Failed",
    uploading: "Uploading",
  };
  const label = labels[status];
  if (!label) return null;
  return (
    <span
      className={`text-[11px] font-medium px-2.5 py-1 rounded-full
        ${styles[status] ?? "bg-sidebar text-muted"}`}
    >
      {label}
    </span>
  );
}
