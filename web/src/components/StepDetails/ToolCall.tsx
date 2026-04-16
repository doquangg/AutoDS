import type { PipelineEvent } from "../../lib/events";

export function ToolCall({ ev }: { ev: PipelineEvent }) {
  return (
    <div className="text-xs font-mono bg-neutral-100 rounded px-2 py-1">
      <div>
        <span className="text-muted">tool:</span> {String(ev.tool)}
      </div>
      {ev.params != null && (
        <div>
          <span className="text-muted">params:</span>{" "}
          {JSON.stringify(ev.params)}
        </div>
      )}
    </div>
  );
}
