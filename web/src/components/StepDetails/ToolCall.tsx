import type { PipelineEvent } from "../../lib/events";

export function ToolCall({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="font-mono text-[11.5px] bg-canvasDeep/60 border border-border
      rounded-lg px-2.5 py-1.5 leading-snug">
      <div className="flex items-baseline gap-2">
        <span className="text-subtle text-[9.5px] uppercase tracking-[0.15em]
          font-semibold">
          tool
        </span>
        <span className="text-ink font-semibold tracking-tight">
          {String(ev.tool)}
        </span>
      </div>
      {ev.params != null && (
        <div className="mt-0.5 text-muted break-all">
          {JSON.stringify(ev.params)}
        </div>
      )}
    </div>
  );
}
