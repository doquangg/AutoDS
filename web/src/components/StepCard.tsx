import { useState } from "react";
import { EventTypes } from "../lib/events";
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

function StatusIcon({ status }: { status: StepCardData["status"] }) {
  if (status === "done") {
    return (
      <span
        className="inline-flex items-center justify-center w-5 h-5 rounded-full
          bg-success/10 text-success"
      >
        <CheckIcon size={12} strokeWidth={3} />
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span
        className="inline-flex items-center justify-center w-5 h-5 rounded-full
          bg-danger/10 text-danger"
      >
        <XIcon size={12} strokeWidth={3} />
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center justify-center w-5 h-5 rounded-full
        bg-running/10 text-running"
    >
      <SpinnerIcon size={12} strokeWidth={2.5} />
    </span>
  );
}

export function StepCard({ step }: { step: StepCardData }) {
  const [open, setOpen] = useState(false);
  const hasDetails = step.details.length > 0;
  const label = NICE_NAME[step.node] ?? step.node;

  return (
    <div className="group animate-fade-in">
      <button
        onClick={() => hasDetails && setOpen(!open)}
        disabled={!hasDetails}
        className={`flex items-center gap-2.5 w-full text-left px-2 py-1.5
          rounded-lg transition ${
            hasDetails
              ? "hover:bg-canvas cursor-pointer"
              : "cursor-default"
          }`}
      >
        <StatusIcon status={step.status} />
        <span
          className={`text-[14px] font-medium ${
            step.status === "running" ? "text-ink" : "text-ink"
          }`}
        >
          {label}
        </span>
        {step.pass > 0 && (
          <span className="text-[11px] font-medium text-muted bg-canvas
            px-1.5 py-0.5 rounded">
            pass {step.pass + 1}
          </span>
        )}
        {step.durationMs != null && (
          <span className="text-[12px] text-muted tabular-nums">
            {(step.durationMs / 1000).toFixed(1)}s
          </span>
        )}
        {hasDetails && (
          <span className="ml-auto text-muted">
            {open ? (
              <ChevronDownIcon size={14} />
            ) : (
              <ChevronRightIcon size={14} />
            )}
          </span>
        )}
      </button>
      {open && hasDetails && (
        <div
          className="ml-7 mt-1 mb-2 pl-3 border-l-2 border-border
            space-y-1.5 animate-fade-in"
        >
          {step.details.map((d, i) => {
            if (d.type === EventTypes.TOOL_CALL)
              return <ToolCall key={i} ev={d} />;
            if (d.type === EventTypes.ROUTE_DECISION)
              return <RouteDecision key={i} ev={d} />;
            if (d.type === EventTypes.INVESTIGATION_FINDINGS)
              return <InvestigationFindings key={i} ev={d} />;
            if (d.type === EventTypes.CLEANING_RECIPE)
              return <CleaningRecipe key={i} ev={d} />;
            if (d.type === EventTypes.MODEL_METADATA)
              return <ModelMetadata key={i} ev={d} />;
            if (d.type === EventTypes.LOG)
              return (
                <div key={i} className="text-[12px] text-muted leading-snug">
                  {String(d.message)}
                </div>
              );
            return null;
          })}
        </div>
      )}
    </div>
  );
}
