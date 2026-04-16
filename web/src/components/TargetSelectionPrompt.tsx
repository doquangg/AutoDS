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
    <div className="self-start max-w-[85%] w-full bg-surface border border-accent/30
      rounded-2xl rounded-bl-sm shadow-card overflow-hidden animate-slide-up">
      <div className="flex items-center gap-2 px-4 py-3 bg-accent/5
        border-b border-accent/15">
        <TargetIcon size={16} className="text-accent" />
        <div className="text-sm font-semibold text-ink">
          Which column should I predict?
        </div>
      </div>
      <div className="p-3 space-y-1">
        {targetPrompt.candidates.map((c) => (
          <button
            key={c.name}
            disabled={submitting}
            onClick={() => pick(c.name)}
            className={`w-full text-left px-3 py-2 rounded-lg border transition
              ${
                chosen === c.name
                  ? "border-accent bg-accent/5"
                  : "border-transparent hover:border-border hover:bg-canvas"
              }
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-[13px] font-semibold text-ink">
                {c.name}
              </span>
            </div>
            <div className="text-[12px] text-muted mt-0.5 leading-snug">
              {c.rationale}
            </div>
          </button>
        ))}
      </div>
      <div className="border-t border-border px-3 py-2.5 bg-canvas flex gap-2
        items-center">
        <select
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          disabled={submitting}
          className="flex-1 text-[13px] border border-border rounded-md
            px-2 py-1.5 bg-surface focus:outline-none focus:ring-2
            focus:ring-accent/30 focus:border-accent disabled:opacity-50"
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
          className="text-[13px] font-medium px-3 py-1.5 rounded-md
            bg-accent text-white hover:bg-accentHover
            disabled:bg-subtle disabled:cursor-not-allowed transition"
        >
          Use
        </button>
      </div>
    </div>
  );
}
