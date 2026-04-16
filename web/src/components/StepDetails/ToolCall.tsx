import type { PipelineEvent } from "../../lib/events";

export function ToolCall({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="text-[12px] font-mono bg-canvas border border-border
      rounded-md px-2.5 py-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-subtle text-[10px] uppercase tracking-wide">
          tool
        </span>
        <span className="text-ink font-medium">{String(ev.tool)}</span>
      </div>
      {ev.params != null && (
        <div className="mt-0.5 text-muted break-all">
          {JSON.stringify(ev.params)}
        </div>
      )}
    </div>
  );
}
