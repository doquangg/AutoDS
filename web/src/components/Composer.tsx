import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { FileChip } from "./FileChip";

interface Props {
  mode: "initial" | "followup";
  disabled?: boolean;
  onSubmitInitial: (file: File, query: string) => void;
  onSubmitFollowup: (question: string) => void;
}

export function Composer({
  mode,
  disabled,
  onSubmitInitial,
  onSubmitFollowup,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith(".csv")) setFile(f);
  }

  function submit() {
    if (mode === "initial") {
      if (!file || !text.trim()) return;
      onSubmitInitial(file, text.trim());
      setFile(null);
      setText("");
    } else {
      if (!text.trim()) return;
      onSubmitFollowup(text.trim());
      setText("");
    }
  }

  const canSubmit =
    mode === "initial" ? !!file && !!text.trim() : !!text.trim();

  return (
    <div
      className="border border-neutral-300 rounded-2xl bg-white p-3 shadow-sm"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      {mode === "initial" && file && (
        <div className="mb-2">
          <FileChip name={file.name} size={file.size} />
        </div>
      )}
      <div className="flex items-end gap-2">
        {mode === "initial" && (
          <button
            className="text-muted hover:text-ink"
            onClick={() => inputRef.current?.click()}
            aria-label="Attach CSV"
          >
            📎
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={1}
          placeholder={
            mode === "initial"
              ? "Drop a CSV and ask a question..."
              : "Ask a follow-up about this run..."
          }
          className="flex-1 resize-none border-0 outline-none bg-transparent placeholder:text-muted"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          disabled={!canSubmit || disabled}
          onClick={submit}
          className="px-3 py-1 rounded-lg bg-accent text-white disabled:opacity-40"
        >
          ↑
        </button>
      </div>
    </div>
  );
}
