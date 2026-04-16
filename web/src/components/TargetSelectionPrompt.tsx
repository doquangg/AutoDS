import { useState } from "react";
import { useStore } from "../store";
import { resumeSession } from "../lib/api";

export function TargetSelectionPrompt() {
  const { sessionId, targetPrompt } = useStore();
  const [submitting, setSubmitting] = useState(false);
  const [custom, setCustom] = useState("");

  if (!targetPrompt || !sessionId) return null;

  async function pick(name: string) {
    setSubmitting(true);
    await resumeSession(sessionId!, name);
    setSubmitting(false);
  }

  return (
    <div className="self-start max-w-[80%] bg-white border border-accent rounded-2xl rounded-bl-md p-3">
      <div className="text-sm mb-2">Which column should I predict?</div>
      <div className="space-y-1">
        {targetPrompt.candidates.map((c) => (
          <button
            key={c.name}
            disabled={submitting}
            onClick={() => pick(c.name)}
            className="w-full text-left text-sm px-2 py-1 rounded hover:bg-neutral-100 disabled:opacity-40"
          >
            <span className="font-medium">{c.name}</span>
            <span className="text-muted ml-2">{c.rationale}</span>
          </button>
        ))}
      </div>
      <div className="mt-2 flex gap-2">
        <select
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          className="text-sm border border-neutral-300 rounded px-2 py-1 flex-1"
        >
          <option value="">Other column...</option>
          {targetPrompt.allColumns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <button
          disabled={!custom || submitting}
          onClick={() => pick(custom)}
          className="text-sm px-2 py-1 rounded bg-accent text-white disabled:opacity-40"
        >
          Use
        </button>
      </div>
    </div>
  );
}
