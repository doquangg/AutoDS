import type { PipelineEvent } from "../../lib/events";

export function RouteDecision({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="text-[12px] leading-snug flex items-center gap-1.5">
      <span className="text-subtle text-[10px] uppercase tracking-wide">
        route
      </span>
      <span className="text-ink font-medium">{String(ev.router)}</span>
      <span className="text-subtle">→</span>
      <span className="text-ink">{String(ev.decision)}</span>
    </div>
  );
}
