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
  variant?: "light" | "dark" | "subtle";
}) {
  const styles: Record<string, string> = {
    light: "bg-canvasDeep text-ink border border-border",
    dark: "bg-ink/85 text-canvas border border-ink",
    subtle: "bg-surface/80 text-inkSoft border border-border/80",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-[11.5px]
        font-medium rounded-full ${styles[variant]}`}
    >
      <FileIcon size={11} className="shrink-0 opacity-70" />
      <span className="truncate max-w-[260px]">{name}</span>
      {size !== undefined && (
        <span className="opacity-60 tabular-nums font-mono text-[10.5px]">
          {fmtSize(size)}
        </span>
      )}
    </span>
  );
}
