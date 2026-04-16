import { useStore } from "../store";
import { UserMessage } from "./UserMessage";

export function ChatThread() {
  const { userQuery, dataset, steps, finalAnswerTokens, qaMessages } =
    useStore();
  const finalAnswer = finalAnswerTokens.join("");

  return (
    <div className="flex flex-col gap-4 px-4 py-6">
      {userQuery && dataset && <UserMessage text={userQuery} file={dataset} />}
      {steps.length > 0 && (
        <div className="self-start max-w-[90%] bg-white border border-neutral-200 rounded-2xl px-4 py-3 shadow-sm">
          {/* Step cards rendered in Task 12 */}
          <div className="space-y-1">
            {steps.map((s) => (
              <div key={s.key} className="text-sm">
                {s.status === "done"
                  ? "✓"
                  : s.status === "failed"
                    ? "✗"
                    : "⟳"}{" "}
                {s.node} (pass {s.pass})
                {s.durationMs ? ` — ${(s.durationMs / 1000).toFixed(1)}s` : ""}
              </div>
            ))}
          </div>
          {finalAnswer && (
            <div className="mt-3 prose prose-sm">{finalAnswer}</div>
          )}
        </div>
      )}
      {qaMessages.map((m, i) =>
        m.role === "user" ? (
          <UserMessage key={i} text={m.text} />
        ) : (
          <div
            key={i}
            className="self-start max-w-[80%] bg-white border border-neutral-200 px-4 py-2 rounded-2xl rounded-bl-md"
          >
            {m.text}
            {m.streaming && <span className="animate-pulse">▍</span>}
          </div>
        ),
      )}
    </div>
  );
}
