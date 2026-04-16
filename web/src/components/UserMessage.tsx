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
      <div className="max-w-[85%] bg-userBubble text-white px-4 py-2.5 rounded-2xl rounded-br-sm shadow-card">
        <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
          {text}
        </div>
        {file && (
          <div className="mt-2">
            <FileChip
              variant="dark"
              name={`${file.filename} · ${file.rows.toLocaleString()} rows × ${file.cols} cols`}
            />
          </div>
        )}
      </div>
    </div>
  );
}
