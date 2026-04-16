import type { PipelineEvent } from "../../lib/events";

export function ModelMetadata({ ev }: { ev: PipelineEvent }) {
  return (
    <details className="text-xs bg-neutral-50 rounded px-2 py-1" open>
      <summary className="cursor-pointer">Model metadata</summary>
      <pre className="mt-1 whitespace-pre-wrap font-mono">
        {String(ev.raw)}
      </pre>
    </details>
  );
}
