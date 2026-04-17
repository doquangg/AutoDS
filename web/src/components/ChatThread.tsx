import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useStore } from "../store";
import type { StepCard as StepCardData } from "../store";
import { UserMessage } from "./UserMessage";
import { StepCard } from "./StepCard";
import { TargetSelectionPrompt } from "./TargetSelectionPrompt";
import {
  CheckIcon,
  ChevronDownIcon,
  LogoMark,
  SpinnerIcon,
  XIcon,
} from "./Icons";

// `tools` is the executor for whatever the investigator just requested,
// so the user perceives investigator+tools as a single "investigation"
// phase. Mapping both nodes to one group key collapses the alternating
// investigator → tools → investigator loop into one card.
function groupKeyOf(node: string): string {
  if (node === "investigator" || node === "tools") return "investigation";
  return node;
}

/**
 * Group consecutive runs that share a group key into a single card.
 * Multi-pass cleaning and the investigator loop produce a lot of
 * same-phase repetition; grouping them keeps the pipeline view scannable
 * while still letting the user drill in per-run.
 *
 * We only collapse *adjacent* steps so ordering is preserved (e.g.
 * profiler-pass-0 and profiler-pass-1 remain separate when other work
 * sits between them).
 */
function groupConsecutive(steps: StepCardData[]): StepCardData[][] {
  const groups: StepCardData[][] = [];
  for (const s of steps) {
    const tail = groups[groups.length - 1];
    if (tail && groupKeyOf(tail[0].node) === groupKeyOf(s.node)) {
      tail.push(s);
    } else {
      groups.push([s]);
    }
  }
  return groups;
}

function AssistantMark() {
  return (
    <div className="shrink-0 mt-0.5 select-none" aria-hidden>
      <LogoMark size={26} />
    </div>
  );
}

function ThinkingHeader({
  label,
  state,
}: {
  label: string;
  state: "running" | "complete" | "failed";
}) {
  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className="uppercase tracking-[0.18em] text-[10.5px] text-subtle
        font-medium">
        Pipeline
      </span>
      <span className="h-px flex-1 bg-border" />
      {state === "running" && (
        <span className="flex items-center gap-1.5 text-running">
          <span className="dot-flash" />
          <span className="dot-flash" />
          <span className="dot-flash" />
          <span className="shimmer-text text-[12px] font-medium tracking-tight
            ml-1">
            {label}
          </span>
        </span>
      )}
      {state === "complete" && (
        <span className="text-success text-[11px] font-medium tracking-wide
          uppercase">
          Done
        </span>
      )}
      {state === "failed" && (
        <span className="text-danger text-[11px] font-medium tracking-wide
          uppercase">
          Failed
        </span>
      )}
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
  const showPipeline = steps.length > 0;
  const isRunning = status === "running" || status === "awaiting_target";
  const pipelineState: "running" | "complete" | "failed" =
    status === "failed"
      ? "failed"
      : status === "complete"
        ? "complete"
        : "running";
  const runningLabel =
    status === "awaiting_target" ? "Awaiting your input" : "Thinking";

  return (
    <div className="flex flex-col gap-7 px-5 pt-6 pb-4 max-w-3xl mx-auto w-full">
      {!userQuery && <EmptyState />}

      {userQuery && dataset && (
        <UserMessage text={userQuery} file={dataset} />
      )}

      {showPipeline && (
        <section className="animate-slide-up">
          <div className="flex items-start gap-3.5">
            <AssistantMark />
            <div className="flex-1 min-w-0">
              <ThinkingHeader
                label={runningLabel}
                state={pipelineState}
              />
              <StepGroups steps={steps} />


              {finalAnswer && (
                <div className="mt-6 pt-5 border-t border-border/70">
                  <div className="prose-answer">
                    <ReactMarkdown>{finalAnswer}</ReactMarkdown>
                    {isRunning && <span className="caret-blink" />}
                  </div>
                </div>
              )}

              {error && status === "failed" && (
                <div className="mt-5 border border-danger/25 bg-danger/5
                  rounded-xl px-4 py-3 text-[13.5px] text-danger leading-relaxed">
                  <div className="font-semibold mb-0.5">Something went wrong</div>
                  {error}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      <TargetSelectionPrompt />

      {qaMessages.map((m, i) =>
        m.role === "user" ? (
          <UserMessage key={i} text={m.text} />
        ) : m.error ? (
          <div key={i} className="flex items-start gap-3.5 animate-slide-up">
            <AssistantMark />
            <div className="flex-1 min-w-0 pt-0.5">
              <div className="border border-danger/25 bg-danger/5 rounded-xl
                px-4 py-3 text-[13.5px] text-danger leading-relaxed">
                <div className="font-semibold mb-0.5">
                  Couldn't answer that
                </div>
                <div className="font-mono text-[12.5px] text-danger/90
                  whitespace-pre-wrap break-words">
                  {m.text}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div key={i} className="flex items-start gap-3.5 animate-slide-up">
            <AssistantMark />
            <div className="flex-1 min-w-0 pt-0.5">
              <div className="prose-answer">
                <ReactMarkdown>{m.text || " "}</ReactMarkdown>
                {m.streaming && <span className="caret-blink" />}
              </div>
            </div>
          </div>
        ),
      )}
    </div>
  );
}

const NODE_LABEL: Record<string, string> = {
  profiler: "Profiling data",
  target_selector: "Selecting target column",
  investigator: "Investigating data",
  tools: "Running investigation tools",
  code_generator: "Generating cleaning code",
  sandbox: "Executing code",
  re_profile: "Re-profiling",
  pass_reset: "Starting next pass",
  feature_engineering: "Feature engineering",
  autogluon: "Training model",
  answer_agent: "Generating answer",
};

function groupByPass(
  steps: StepCardData[],
): { pass: number; steps: StepCardData[] }[] {
  const groups: { pass: number; steps: StepCardData[] }[] = [];
  for (const s of steps) {
    const tail = groups[groups.length - 1];
    if (tail && tail.pass === s.pass) {
      tail.steps.push(s);
    } else {
      groups.push({ pass: s.pass, steps: [s] });
    }
  }
  return groups;
}

function aggregatePassStatus(
  steps: StepCardData[],
): "running" | "done" | "failed" {
  if (steps.some((s) => s.status === "failed")) return "failed";
  if (steps.some((s) => s.status === "running")) return "running";
  return "done";
}

function PassStatusIcon({
  status,
}: {
  status: "running" | "done" | "failed";
}) {
  if (status === "done") {
    return (
      <span className="inline-flex items-center justify-center w-[20px] h-[20px]
        rounded-full bg-success/12 text-success ring-1 ring-success/20">
        <CheckIcon size={12} strokeWidth={3} />
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex items-center justify-center w-[20px] h-[20px]
        rounded-full bg-danger/12 text-danger ring-1 ring-danger/20">
        <XIcon size={12} strokeWidth={3} />
      </span>
    );
  }
  return (
    <span className="inline-flex items-center justify-center w-[20px] h-[20px]
      rounded-full bg-running/12 text-running ring-1 ring-running/25">
      <SpinnerIcon size={12} strokeWidth={2.6} />
    </span>
  );
}

function StepGroups({ steps }: { steps: StepCardData[] }) {
  const passGroups = useMemo(() => groupByPass(steps), [steps]);
  const latestPass = passGroups[passGroups.length - 1]?.pass ?? null;
  return (
    <div className="mt-3 space-y-1.5">
      {passGroups.map((pg) => (
        <PassDisclosure
          key={pg.pass}
          pass={pg.pass}
          passSteps={pg.steps}
          isLatest={pg.pass === latestPass}
        />
      ))}
    </div>
  );
}

function PassDisclosure({
  pass,
  passSteps,
  isLatest,
}: {
  pass: number;
  passSteps: StepCardData[];
  isLatest: boolean;
}) {
  // Latest pass opens automatically; when a new pass starts, the old one
  // auto-collapses. Users can still toggle mid-pass — we reset the override
  // whenever latest-ness changes so predictable accordion behavior wins.
  const [override, setOverride] = useState<boolean | null>(null);
  useEffect(() => {
    setOverride(null);
  }, [isLatest]);
  const open = override ?? isLatest;

  const status = aggregatePassStatus(passSteps);
  const nodeGroups = useMemo(() => groupConsecutive(passSteps), [passSteps]);

  const totalDuration = passSteps.every((s) => s.durationMs != null)
    ? passSteps.reduce((t, s) => t + (s.durationMs ?? 0), 0)
    : undefined;

  const runningStep = passSteps.find((s) => s.status === "running");
  const runningLabel = runningStep
    ? (NODE_LABEL[runningStep.node] ?? runningStep.node)
    : null;

  const stepCount = passSteps.length;
  const stepCountLabel = `${stepCount} ${stepCount === 1 ? "step" : "steps"}`;

  return (
    <div
      className={`rounded-xl border transition-colors
        ${status === "running"
          ? "border-running/30 bg-running/[0.03]"
          : status === "failed"
            ? "border-danger/25 bg-danger/[0.03]"
            : "border-border/60 bg-surface/40"}
        ${open ? "shadow-soft" : ""}`}
    >
      <button
        onClick={() => setOverride(!open)}
        className="flex items-center gap-3 w-full text-left px-3 py-2
          rounded-xl hover:bg-canvasDeep/40 transition"
      >
        <PassStatusIcon status={status} />
        <span className="font-display text-[14.5px] tracking-tight text-ink">
          Pass <span className="tabular-nums">{pass + 1}</span>
        </span>
        <span className="text-[11px] text-subtle tabular-nums">
          {stepCountLabel}
        </span>
        {!open && runningLabel && (
          <span className="min-w-0 truncate text-[11.5px] text-muted italic">
            · {runningLabel}…
          </span>
        )}
        {totalDuration != null && (
          <span className="ml-auto text-[11.5px] text-subtle tabular-nums
            font-mono">
            {(totalDuration / 1000).toFixed(1)}s
          </span>
        )}
        <span
          className={`${totalDuration != null ? "" : "ml-auto"} text-subtle
            transition-transform duration-200
            ${open ? "rotate-0" : "-rotate-90"}`}
        >
          <ChevronDownIcon size={14} />
        </span>
      </button>

      {open && (
        <div className="px-2 pb-1.5 pt-0.5 space-y-0.5 animate-fade-in">
          {nodeGroups.map((g) => (
            <StepCard key={g[0].key} steps={g} />
          ))}
        </div>
      )}
    </div>
  );
}

const EXAMPLE_PROMPTS = [
  {
    headline: "Predict customer churn",
    body: "from a subscription dataset and explain which signals matter most.",
  },
  {
    headline: "Score housing listings",
    body: "against a target price and surface the worst outliers.",
  },
  {
    headline: "Cluster transactions",
    body: "to find suspicious patterns worth a closer look.",
  },
];

function EmptyState() {
  return (
    <div className="flex flex-col items-center text-center pt-16 pb-6
      animate-fade-in">
      <div className="relative mb-7">
        <div className="absolute inset-0 blur-2xl opacity-50
          bg-accent/25 rounded-full scale-110" aria-hidden />
        <div className="relative">
          <LogoMark size={56} />
        </div>
      </div>

      <h1 className="font-display text-[54px] leading-[1.02] tracking-tight
        text-ink">
        What would you like
        <br />
        to <em className="italic text-accent">understand</em> today?
      </h1>
      <p className="text-muted mt-5 max-w-md text-[15px] leading-relaxed">
        Drop a CSV below and ask anything. AutoDS profiles, cleans, models, and
        answers — showing its work at every step.
      </p>

      <div className="mt-10 grid sm:grid-cols-3 gap-2.5 w-full max-w-2xl
        text-left stagger">
        {EXAMPLE_PROMPTS.map((p) => (
          <div
            key={p.headline}
            className="group px-4 py-3.5 rounded-xl border border-border/70
              bg-surface/60 hover:bg-surface hover:border-borderStrong
              transition shadow-soft cursor-default"
          >
            <div className="font-display text-[17px] leading-tight text-ink
              tracking-tight mb-1">
              {p.headline}
            </div>
            <div className="text-[12.5px] text-muted leading-snug">
              {p.body}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
