import type { PipelineEvent } from "../../lib/events";

export function RouteDecision({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="text-[11.5px] leading-snug flex items-center gap-1.5
      font-mono">
      <span className="text-subtle text-[9.5px] uppercase tracking-[0.15em]
        font-semibold">
        route
      </span>
      <span className="text-ink font-semibold tracking-tight">
        {String(ev.router)}
      </span>
      <span className="text-subtle">→</span>
      <span className="text-inkSoft">{String(ev.decision)}</span>
    </div>
  );
}
