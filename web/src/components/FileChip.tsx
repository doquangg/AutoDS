import { FileIcon } from "./Icons";

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function FileChip({
  name,
  size,
  variant = "light",
}: {
  name: string;
  size?: number;
  variant?: "light" | "dark";
}) {
  const styles =
    variant === "dark"
      ? "bg-white/20 text-white"
      : "bg-canvas text-ink border border-border";
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 text-[11px] font-medium rounded-md ${styles}`}
    >
      <FileIcon size={12} className="shrink-0 opacity-80" />
      <span className="truncate max-w-[240px]">{name}</span>
      {size !== undefined && (
        <span className="opacity-70 tabular-nums">{fmtSize(size)}</span>
      )}
    </span>
  );
}
