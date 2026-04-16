import { FileChip } from "./FileChip";

export function UserMessage({
  text,
  file,
}: {
  text: string;
  file?: { filename: string; rows: number; cols: number };
}) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="max-w-[82%] bg-userBubble text-ink
        px-4 py-3 rounded-[20px] rounded-br-[6px] border border-border/60
        shadow-soft">
        <div className="text-[15px] leading-relaxed whitespace-pre-wrap
          tracking-[-0.005em]">
          {text}
        </div>
        {file && (
          <div className="mt-2.5">
            <FileChip
              variant="subtle"
              name={`${file.filename} · ${file.rows.toLocaleString()} rows × ${file.cols} cols`}
            />
          </div>
        )}
      </div>
    </div>
  );
}
