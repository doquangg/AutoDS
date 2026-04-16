import type { PipelineEvent } from "../../lib/events";

export function ModelMetadata({ ev }: { ev: PipelineEvent }) {
  return (
    <details
      className="text-[12px] bg-canvas border border-border rounded-md"
      open
    >
      <summary className="cursor-pointer px-2.5 py-1.5 font-medium text-ink
        hover:bg-border/40 rounded-md">
        Model metadata
      </summary>
      <pre className="px-2.5 pb-2.5 whitespace-pre-wrap font-mono text-[11px]
        text-muted leading-relaxed">
        {String(ev.raw)}
      </pre>
    </details>
  );
}
