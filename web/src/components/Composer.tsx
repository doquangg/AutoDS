import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { FileChip } from "./FileChip";
import { PaperclipIcon, SendIcon, XIcon } from "./Icons";

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
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [text]);

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (mode !== "initial") return;
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith(".csv")) setFile(f);
  }

  function submit() {
    if (disabled) return;
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
      className={`relative border rounded-2xl bg-surface shadow-card transition
        ${dragOver ? "border-accent ring-2 ring-accent/20" : "border-border"}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (mode === "initial") setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {mode === "initial" && file && (
        <div className="px-3 pt-3">
          <div className="inline-flex items-center gap-1.5">
            <FileChip name={file.name} size={file.size} />
            <button
              onClick={() => setFile(null)}
              className="p-0.5 rounded hover:bg-border text-muted hover:text-ink transition"
              aria-label="Remove file"
            >
              <XIcon size={12} />
            </button>
          </div>
        </div>
      )}
      <div className="flex items-end gap-2 px-3 py-2.5">
        {mode === "initial" && (
          <button
            className="p-1.5 rounded-lg text-muted hover:text-ink hover:bg-canvas transition"
            onClick={() => inputRef.current?.click()}
            aria-label="Attach CSV"
            title="Attach CSV"
          >
            <PaperclipIcon size={18} />
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
          ref={taRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={1}
          placeholder={
            mode === "initial"
              ? "Drop a CSV and ask a question…"
              : "Ask a follow-up about this run…"
          }
          className="flex-1 resize-none border-0 outline-none bg-transparent
            placeholder:text-subtle text-[15px] leading-relaxed py-1.5
            max-h-[200px]"
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
          className="shrink-0 p-2 rounded-xl bg-accent text-white
            hover:bg-accentHover disabled:bg-subtle disabled:cursor-not-allowed
            transition shadow-sm"
          aria-label="Send"
        >
          <SendIcon size={16} />
        </button>
      </div>
      {dragOver && (
        <div className="pointer-events-none absolute inset-0 rounded-2xl
          bg-accent/5 flex items-center justify-center">
          <div className="text-sm font-medium text-accent">
            Drop CSV to attach
          </div>
        </div>
      )}
    </div>
  );
}
