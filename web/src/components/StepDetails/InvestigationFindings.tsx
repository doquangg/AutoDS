import type { PipelineEvent } from "../../lib/events";

export function InvestigationFindings({ ev }: { ev: PipelineEvent }) {
  return (
    <details className="text-xs bg-neutral-50 rounded px-2 py-1">
      <summary className="cursor-pointer">Investigation findings</summary>
      <pre className="mt-1 whitespace-pre-wrap font-mono">
        {String(ev.raw)}
      </pre>
    </details>
  );
}
