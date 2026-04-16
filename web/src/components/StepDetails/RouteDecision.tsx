import type { PipelineEvent } from "../../lib/events";

export function RouteDecision({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="text-xs">
      <span className="text-muted">route:</span> {String(ev.router)} →{" "}
      {String(ev.decision)}
    </div>
  );
}
