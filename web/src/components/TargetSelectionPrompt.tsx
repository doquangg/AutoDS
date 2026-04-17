import { useState } from "react";
import { useStore } from "../store";
import { resumeSession } from "../lib/api";
import { TargetIcon } from "./Icons";

export function TargetSelectionPrompt() {
  const sessionId = useStore((s) => s.sessionId);
  const targetPrompt = useStore((s) => s.targetPrompt);
  const [submitting, setSubmitting] = useState(false);
  const [chosen, setChosen] = useState<string | null>(null);
  const [custom, setCustom] = useState("");

  if (!targetPrompt || !sessionId) return null;

  async function pick(name: string) {
    setChosen(name);
    setSubmitting(true);
    try {
      await resumeSession(sessionId!, name);
    } catch (e) {
      setSubmitting(false);
      setChosen(null);
      console.error(e);
    }
  }

  return (
    <div className="ml-[38px] mt-1 max-w-[640px] animate-slide-up">
      <div className="bg-surface/80 border border-accent/20 rounded-2xl
        shadow-soft overflow-hidden backdrop-blur-sm">
        <div className="flex items-center gap-2.5 px-4 py-3
          bg-accent/[0.05] border-b border-accent/15">
          <span className="w-7 h-7 rounded-full bg-accent/10 text-accent
            flex items-center justify-center">
            <TargetIcon size={14} />
          </span>
          <div>
            <div className="text-[13.5px] font-semibold text-ink
              tracking-[-0.005em]">
              Which column should I predict?
            </div>
            <div className="text-[11.5px] text-muted mt-0.5">
              I surfaced a few candidates — pick one or choose your own.
            </div>
          </div>
        </div>

        <div className="p-2 space-y-0.5">
          {targetPrompt.candidates.map((c) => (
            <button
              key={c.name}
              disabled={submitting}
              onClick={() => pick(c.name)}
              className={`w-full text-left px-3 py-2.5 rounded-xl transition
                border
                ${
                  chosen === c.name
                    ? "border-accent bg-accent/5"
                    : "border-transparent hover:border-border hover:bg-canvasDeep/60"
                }
                disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[13px] font-semibold text-ink
                  tracking-tight">
                  {c.name}
                </span>
              </div>
              <div className="text-[12.5px] text-muted mt-0.5 leading-snug">
                {c.rationale}
              </div>
            </button>
          ))}
        </div>

        <div className="border-t border-border px-3 py-2.5 bg-canvasDeep/40
          flex gap-2 items-center">
          <select
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            disabled={submitting}
            className="flex-1 text-[13px] border border-border rounded-lg
              px-2.5 py-1.5 bg-surface focus:outline-none
              focus:ring-2 focus:ring-accent/25 focus:border-accent/60
              disabled:opacity-50 font-mono tracking-tight text-ink"
          >
            <option value="">Other column…</option>
            {targetPrompt.allColumns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <button
            disabled={!custom || submitting}
            onClick={() => pick(custom)}
            className="text-[13px] font-medium px-3.5 py-1.5 rounded-lg
              bg-ink text-canvas hover:bg-accent
              disabled:bg-border disabled:text-subtle disabled:cursor-not-allowed
              transition-colors"
          >
            Use
          </button>
        </div>
      </div>
    </div>
  );
}
