import { FileChip } from "./FileChip";

export function UserMessage({
  text,
  file,
}: {
  text: string;
  file?: { filename: string; rows: number; cols: number };
}) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] bg-accent text-white px-4 py-2 rounded-2xl rounded-br-md">
        <div>{text}</div>
        {file && (
          <div className="mt-1 text-xs opacity-90">
            <FileChip
              name={`${file.filename} (${file.rows} rows × ${file.cols} cols)`}
            />
          </div>
        )}
      </div>
    </div>
  );
}
