import { useEffect } from "react";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
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

  async function handleInitial(file: File, query: string) {
    const r = await createSession(file, query);
    startSession({
      sessionId: r.session_id,
      filename: r.filename,
      rows: r.rows,
      cols: r.cols,
      userQuery: query,
    });
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
        const ev = JSON.parse(m[1]);
        if (ev.type === "qa_token") appendQAToken(ev.text);
        else if (ev.type === "qa_complete") finishQA();
      }
    }
  }

  return (
    <div className="h-full flex flex-col max-w-3xl mx-auto">
      <header className="px-4 py-3 border-b border-neutral-200 flex items-center justify-between">
        <div className="font-semibold">AutoDS</div>
        {sessionId && (
          <button
            onClick={reset}
            className="text-sm text-muted hover:text-ink"
          >
            New session
          </button>
        )}
      </header>
      <div className="flex-1 overflow-y-auto">
        <ChatThread />
      </div>
      <div className="p-4">
        <Composer
          mode={status === "complete" ? "followup" : "initial"}
          disabled={status === "running" || status === "awaiting_target"}
          onSubmitInitial={handleInitial}
          onSubmitFollowup={handleFollowup}
        />
      </div>
    </div>
  );
}
