import { useState } from "react";
import { EventTypes } from "../lib/events";
import type { PipelineEvent } from "../lib/events";
import type { StepCard as StepCardData } from "../store";
import { ToolCall } from "./StepDetails/ToolCall";
import { RouteDecision } from "./StepDetails/RouteDecision";
import { InvestigationFindings } from "./StepDetails/InvestigationFindings";
import { CleaningRecipe } from "./StepDetails/CleaningRecipe";
import { ModelMetadata } from "./StepDetails/ModelMetadata";
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  SpinnerIcon,
  XIcon,
} from "./Icons";

const NICE_NAME: Record<string, string> = {
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

type AggStatus = StepCardData["status"];

function aggregateStatus(steps: StepCardData[]): AggStatus {
  if (steps.some((s) => s.status === "failed")) return "failed";
  if (steps.some((s) => s.status === "running")) return "running";
  return "done";
}

function aggregateDuration(steps: StepCardData[]): number | undefined {
  if (steps.every((s) => s.durationMs != null)) {
    return steps.reduce((t, s) => t + (s.durationMs ?? 0), 0);
  }
  return undefined;
}

function StatusIcon({ status }: { status: AggStatus }) {
  if (status === "done") {
    return (
      <span className="inline-flex items-center justify-center w-[18px] h-[18px]
        rounded-full bg-success/12 text-success ring-1 ring-success/20">
        <CheckIcon size={11} strokeWidth={3} />
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex items-center justify-center w-[18px] h-[18px]
        rounded-full bg-danger/12 text-danger ring-1 ring-danger/20">
        <XIcon size={11} strokeWidth={3} />
      </span>
    );
  }
  return (
    <span className="inline-flex items-center justify-center w-[18px] h-[18px]
      rounded-full bg-running/12 text-running ring-1 ring-running/25">
      <SpinnerIcon size={11} strokeWidth={2.6} />
    </span>
  );
}

function DetailItem({ d }: { d: PipelineEvent }) {
  if (d.type === EventTypes.TOOL_CALL) return <ToolCall ev={d} />;
  if (d.type === EventTypes.ROUTE_DECISION) return <RouteDecision ev={d} />;
  if (d.type === EventTypes.INVESTIGATION_FINDINGS)
    return <InvestigationFindings ev={d} />;
  if (d.type === EventTypes.CLEANING_RECIPE) return <CleaningRecipe ev={d} />;
  if (d.type === EventTypes.MODEL_METADATA) return <ModelMetadata ev={d} />;
  if (d.type === EventTypes.LOG) {
    return (
      <div className="text-[11.5px] text-muted leading-snug font-mono">
        {String(d.message)}
      </div>
    );
  }
  return null;
}

/**
 * A StepCard now represents a group of one-or-more consecutive pipeline
 * runs for the same node. The investigator loop and multi-pass cleaning
 * produce runs of identical nodes (e.g. `investigator, tools, investigator,
 * tools, ...`); we collapse the whole run into a single card whose header
 * shows a count ("5×") and whose expansion lists each individual run with
 * its details.
 */
export function StepCard({ steps }: { steps: StepCardData[] }) {
  const [open, setOpen] = useState(false);
  if (steps.length === 0) return null;

  const first = steps[0];
  const status = aggregateStatus(steps);
  const totalDuration = aggregateDuration(steps);
  const label = NICE_NAME[first.node] ?? first.node;
  const isRunning = status === "running";

  const totalDetails = steps.reduce((n, s) => n + s.details.length, 0);
  const isMulti = steps.length > 1;
  const hasExpandable = totalDetails > 0 || isMulti;

  // Pass range summary: "pass 1" for one, "passes 1–3" for a range.
  const minPass = Math.min(...steps.map((s) => s.pass));
  const maxPass = Math.max(...steps.map((s) => s.pass));
  const passLabel =
    maxPass === 0
      ? null
      : minPass === maxPass
        ? `pass ${minPass + 1}`
        : `passes ${minPass + 1}–${maxPass + 1}`;

  return (
    <div className="group animate-fade-in">
      <button
        onClick={() => hasExpandable && setOpen(!open)}
        disabled={!hasExpandable}
        className={`flex items-center gap-2.5 w-full text-left
          px-2 py-1.5 rounded-lg transition
          ${hasExpandable ? "hover:bg-canvasDeep cursor-pointer" : "cursor-default"}`}
      >
        <StatusIcon status={status} />
        <span
          className={`text-[13.5px] font-medium tracking-[-0.005em] ${
            isRunning ? "shimmer-text" : "text-ink"
          }`}
        >
          {label}
        </span>
        {isMulti && (
          <span className="text-[10.5px] font-semibold text-accent
            bg-accent/10 ring-1 ring-accent/15 px-1.5 py-0.5 rounded-full
            tracking-wide font-mono tabular-nums">
            {steps.length}×
          </span>
        )}
        {passLabel && (
          <span className="text-[10.5px] font-medium text-muted bg-canvasDeep
            px-1.5 py-0.5 rounded tracking-wide font-mono">
            {passLabel}
          </span>
        )}
        {totalDuration != null && (
          <span className="text-[11.5px] text-subtle tabular-nums font-mono">
            {(totalDuration / 1000).toFixed(1)}s
          </span>
        )}
        {hasExpandable && (
          <span className="ml-auto text-subtle group-hover:text-muted
            transition-colors">
            {open ? (
              <ChevronDownIcon size={13} />
            ) : (
              <ChevronRightIcon size={13} />
            )}
          </span>
        )}
      </button>

      {open && hasExpandable && (
        <div className="ml-[22px] mt-1 mb-2 pl-3.5 border-l border-border
          space-y-3 animate-fade-in">
          {isMulti
            ? steps.map((s, runIdx) => (
                <RunBlock
                  key={s.key}
                  step={s}
                  index={runIdx}
                  total={steps.length}
                />
              ))
            : (
              <div className="space-y-1.5">
                {first.details.map((d, i) => (
                  <DetailItem key={i} d={d} />
                ))}
              </div>
            )}
        </div>
      )}
    </div>
  );
}

function RunBlock({
  step,
  index,
  total,
}: {
  step: StepCardData;
  index: number;
  total: number;
}) {
  const passLabel =
    step.pass > 0 ? ` · pass ${step.pass + 1}` : "";
  return (
    <div>
      <div className="flex items-center gap-2 text-[10.5px] uppercase
        tracking-[0.15em] font-semibold text-subtle mb-1.5">
        <span>Run {index + 1} of {total}{passLabel}</span>
        <span className="flex-1 h-px bg-border/80" />
        <span className="text-muted normal-case tracking-normal font-medium">
          {step.status === "done"
            ? "done"
            : step.status === "failed"
              ? "failed"
              : "running"}
          {step.durationMs != null && (
            <span className="ml-1 font-mono tabular-nums text-subtle">
              {(step.durationMs / 1000).toFixed(1)}s
            </span>
          )}
        </span>
      </div>
      {step.details.length > 0 ? (
        <div className="space-y-1.5">
          {step.details.map((d, i) => (
            <DetailItem key={i} d={d} />
          ))}
        </div>
      ) : (
        <div className="text-[11.5px] text-subtle italic">no details</div>
      )}
    </div>
  );
}
