export function FileChip({ name, size }: { name: string; size?: number }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full bg-neutral-200 text-ink">
      <span>{name}</span>
      {size !== undefined && (
        <span className="text-muted">({size.toLocaleString()} B)</span>
      )}
    </span>
  );
}
