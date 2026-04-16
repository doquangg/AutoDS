import { useState } from "react";
import { EventTypes } from "../lib/events";
import type { StepCard as StepCardData } from "../store";
import { ToolCall } from "./StepDetails/ToolCall";
import { RouteDecision } from "./StepDetails/RouteDecision";
import { InvestigationFindings } from "./StepDetails/InvestigationFindings";
import { CleaningRecipe } from "./StepDetails/CleaningRecipe";
import { ModelMetadata } from "./StepDetails/ModelMetadata";

const ICON: Record<StepCardData["status"], string> = {
  running: "⟳",
  done: "✓",
  failed: "✗",
};
const COLOR: Record<StepCardData["status"], string> = {
  running: "text-running animate-spin-slow inline-block",
  done: "text-success",
  failed: "text-danger",
};

const NICE_NAME: Record<string, string> = {
  profiler: "Profiling data",
  target_selector: "Selecting target column",
  investigator: "Investigating data",
  tools: "Running investigation tools",
  code_generator: "Generating cleaning code",
  sandbox: "Sandbox execution",
  re_profile: "Re-profiling",
  pass_reset: "Starting next pass",
  feature_engineering: "Feature engineering",
  autogluon: "Training model",
  answer_agent: "Generating answer",
};

export function StepCard({ step }: { step: StepCardData }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-sm">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left hover:bg-neutral-50 rounded px-1"
      >
        <span className={COLOR[step.status]}>{ICON[step.status]}</span>
        <span>{NICE_NAME[step.node] ?? step.node}</span>
        {step.durationMs != null && (
          <span className="text-muted text-xs">
            ({(step.durationMs / 1000).toFixed(1)}s)
          </span>
        )}
        {step.details.length > 0 && (
          <span className="text-muted text-xs ml-auto">
            {open ? "▾" : "▸"}
          </span>
        )}
      </button>
      {open && step.details.length > 0 && (
        <div className="ml-6 mt-1 space-y-1">
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
                <div key={i} className="text-xs text-muted">
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
