import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { FileChip } from "./FileChip";
import { ArrowUpIcon, PaperclipIcon, XIcon } from "./Icons";

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

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
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
      className={`relative bg-surface shadow-composer rounded-[28px] transition
        ${
          dragOver
            ? "ring-2 ring-accent/35 border-accent"
            : "ring-1 ring-border hover:ring-borderStrong"
        }
        ${disabled ? "opacity-80" : ""}
      `}
      onDragOver={(e) => {
        e.preventDefault();
        if (mode === "initial") setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {mode === "initial" && file && (
        <div className="px-5 pt-3.5 pb-0.5">
          <div className="inline-flex items-center gap-1.5 animate-fade-in">
            <FileChip name={file.name} size={file.size} />
            <button
              onClick={() => setFile(null)}
              className="p-1 rounded-full hover:bg-canvasDeep text-muted
                hover:text-ink transition"
              aria-label="Remove file"
            >
              <XIcon size={12} />
            </button>
          </div>
        </div>
      )}

      <textarea
        ref={taRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={1}
        placeholder={
          mode === "initial"
            ? "Drop a CSV and ask anything…"
            : "Ask a follow-up about this run…"
        }
        className="block w-full resize-none bg-transparent border-0 outline-none
          placeholder:text-subtle text-[15.5px] leading-relaxed
          px-5 pt-4 pb-2 max-h-[220px] tracking-[-0.005em]"
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />

      <div className="flex items-center justify-between gap-2 px-3 pb-3 pt-1">
        <div className="flex items-center gap-1">
          {mode === "initial" && (
            <button
              className="flex items-center gap-1.5 pl-2 pr-3 py-1.5 rounded-full
                text-[12.5px] font-medium text-inkSoft hover:text-ink
                hover:bg-canvasDeep transition"
              onClick={() => inputRef.current?.click()}
              aria-label="Attach CSV"
              title="Attach CSV"
            >
              <PaperclipIcon size={14} />
              <span>Attach CSV</span>
            </button>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden sm:inline text-[11px] text-subtle
            tracking-wide">
            {canSubmit && !disabled ? (
              <>
                <kbd className="font-mono bg-canvasDeep text-muted border
                  border-border rounded px-1 py-0.5 text-[10.5px]">↵</kbd> to send
              </>
            ) : null}
          </span>
          <button
            disabled={!canSubmit || disabled}
            onClick={submit}
            className="shrink-0 w-9 h-9 rounded-full
              flex items-center justify-center
              bg-ink text-canvas hover:bg-accent
              disabled:bg-border disabled:text-subtle disabled:cursor-not-allowed
              transition-colors shadow-pill"
            aria-label="Send"
          >
            <ArrowUpIcon size={16} strokeWidth={2.4} />
          </button>
        </div>
      </div>

      {dragOver && (
        <div className="pointer-events-none absolute inset-0 rounded-[28px]
          bg-accent/5 flex items-center justify-center">
          <div className="text-[13px] font-medium text-accent
            bg-surface/90 px-3 py-1.5 rounded-full border border-accent/30
            shadow-soft">
            Drop CSV to attach
          </div>
        </div>
      )}
    </div>
  );
}
