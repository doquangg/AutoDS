import type { PipelineEvent } from "../../lib/events";

export function InvestigationFindings({ ev }: { ev: PipelineEvent }) {
  return (
    <details className="text-[11.5px] bg-canvasDeep/60 border border-border
      rounded-lg group">
      <summary className="cursor-pointer px-2.5 py-1.5 font-medium text-ink
        hover:bg-canvasDeep rounded-lg list-none select-none
        flex items-center gap-1.5">
        <span className="text-subtle text-[9.5px] uppercase tracking-[0.15em]
          font-semibold">
          findings
        </span>
        <span className="text-inkSoft">Investigation notes</span>
      </summary>
      <pre className="px-2.5 pb-2.5 pt-1 whitespace-pre-wrap font-mono text-[11px]
        text-muted leading-relaxed">
        {String(ev.raw)}
      </pre>
    </details>
  );
}
