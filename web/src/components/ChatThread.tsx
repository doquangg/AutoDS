import ReactMarkdown from "react-markdown";
import { useStore } from "../store";
import { UserMessage } from "./UserMessage";
import { StepCard } from "./StepCard";
import { TargetSelectionPrompt } from "./TargetSelectionPrompt";
import { SparklesIcon } from "./Icons";

function AssistantAvatar() {
  return (
    <div
      className="shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-accent
        to-accentHover text-white flex items-center justify-center
        shadow-sm"
    >
      <SparklesIcon size={14} />
    </div>
  );
}

function RunningIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-[13px] text-muted">
      <span className="text-running">
        <span className="dot-flash" />
        <span className="dot-flash" />
        <span className="dot-flash" />
      </span>
      <span>{label}</span>
    </div>
  );
}

export function ChatThread() {
  const userQuery = useStore((s) => s.userQuery);
  const dataset = useStore((s) => s.dataset);
  const steps = useStore((s) => s.steps);
  const finalAnswerTokens = useStore((s) => s.finalAnswerTokens);
  const qaMessages = useStore((s) => s.qaMessages);
  const status = useStore((s) => s.status);
  const error = useStore((s) => s.error);

  const finalAnswer = finalAnswerTokens.join("");
  const showPipelineCard = steps.length > 0;
  const isRunning = status === "running" || status === "awaiting_target";

  return (
    <div className="flex flex-col gap-5 px-4 py-6 max-w-3xl mx-auto w-full">
      {!userQuery && (
        <EmptyState />
      )}

      {userQuery && dataset && (
        <UserMessage text={userQuery} file={dataset} />
      )}

      {showPipelineCard && (
        <div className="flex items-start gap-3 animate-slide-up">
          <AssistantAvatar />
          <div
            className="flex-1 bg-surface border border-border rounded-2xl
              rounded-tl-sm shadow-card overflow-hidden"
          >
            <div className="px-4 py-3 border-b border-border flex items-center
              justify-between">
              <div className="text-[13px] font-semibold text-ink">
                Pipeline
              </div>
              {isRunning && (
                <RunningIndicator
                  label={
                    status === "awaiting_target"
                      ? "awaiting your input"
                      : "running"
                  }
                />
              )}
              {status === "complete" && (
                <div className="text-[12px] font-medium text-success">
                  Complete
                </div>
              )}
              {status === "failed" && (
                <div className="text-[12px] font-medium text-danger">
                  Failed
                </div>
              )}
            </div>
            <div className="p-2 space-y-0.5">
              {steps.map((s) => (
                <StepCard key={s.key} step={s} />
              ))}
            </div>
            {finalAnswer && (
              <div className="border-t border-border px-4 py-3 bg-canvas/50">
                <div className="text-[11px] font-semibold uppercase
                  tracking-wide text-muted mb-1.5">
                  Answer
                </div>
                <div className="prose-answer">
                  <ReactMarkdown>{finalAnswer}</ReactMarkdown>
                </div>
              </div>
            )}
            {error && status === "failed" && (
              <div className="border-t border-danger/20 px-4 py-3 bg-danger/5
                text-[13px] text-danger">
                {error}
              </div>
            )}
          </div>
        </div>
      )}

      <TargetSelectionPrompt />

      {qaMessages.map((m, i) =>
        m.role === "user" ? (
          <UserMessage key={i} text={m.text} />
        ) : (
          <div key={i} className="flex items-start gap-3 animate-slide-up">
            <AssistantAvatar />
            <div
              className="flex-1 max-w-[85%] bg-surface border border-border
                px-4 py-3 rounded-2xl rounded-tl-sm shadow-card"
            >
              <div className="prose-answer">
                <ReactMarkdown>{m.text || " "}</ReactMarkdown>
              </div>
              {m.streaming && (
                <span className="inline-block w-1.5 h-4 ml-0.5 bg-accent
                  animate-pulse rounded-sm align-middle" />
              )}
            </div>
          </div>
        ),
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center
      animate-fade-in">
      <div
        className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent
          to-accentHover text-white flex items-center justify-center
          shadow-card mb-4"
      >
        <SparklesIcon size={24} />
      </div>
      <h1 className="text-2xl font-semibold text-ink tracking-tight">
        AutoDS
      </h1>
      <p className="text-muted mt-2 max-w-sm text-[15px] leading-relaxed">
        Drop a CSV and ask a data-science question. The pipeline profiles,
        cleans, models, and answers — transparently.
      </p>
    </div>
  );
}
