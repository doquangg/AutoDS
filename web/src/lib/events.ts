// Mirrors core/web/events.py — keep in sync.
export const EventTypes = {
  SESSION_STARTED: "session_started",
  NODE_STARTED: "node_started",
  NODE_COMPLETED: "node_completed",
  NODE_FAILED: "node_failed",
  ROUTE_DECISION: "route_decision",
  TOOL_CALL: "tool_call",
  INVESTIGATION_FINDINGS: "investigation_findings",
  CLEANING_RECIPE: "cleaning_recipe",
  PROFILE_SUMMARY: "profile_summary",
  MODEL_METADATA: "model_metadata",
  TARGET_SELECTION_REQUIRED: "target_selection_required",
  TARGET_SELECTION_RESOLVED: "target_selection_resolved",
  FINAL_ANSWER_TOKEN: "final_answer_token",
  PIPELINE_COMPLETE: "pipeline_complete",
  PIPELINE_FAILED: "pipeline_failed",
  QA_TOKEN: "qa_token",
  QA_COMPLETE: "qa_complete",
  QA_ERROR: "qa_error",
  HEARTBEAT: "heartbeat",
  REPLAY_TRUNCATED: "replay_truncated",
  LOG: "log",
} as const;

export type EventType = (typeof EventTypes)[keyof typeof EventTypes];

export interface PipelineEvent {
  type: EventType;
  seq: number;
  ts: string;
  [k: string]: unknown;
}

export interface TargetCandidate {
  name: string;
  rationale: string;
}
