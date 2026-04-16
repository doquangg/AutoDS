import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

const base = (size = 16) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export function CheckIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

export function XIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  );
}

export function SpinnerIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest} className={`${rest.className ?? ""} animate-spin-slow`}>
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

export function ChevronRightIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

export function ChevronDownIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

export function PaperclipIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.64 17.18a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

export function SendIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M22 2L11 13" />
      <path d="M22 2l-7 20-4-9-9-4 20-7z" />
    </svg>
  );
}

export function SparklesIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6L12 3z" />
      <path d="M19 3v4M21 5h-4M5 17v4M7 19H3" />
    </svg>
  );
}

export function FileIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </svg>
  );
}

export function RefreshIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <path d="M23 4v6h-6M1 20v-6h6" />
      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
  );
}

export function TargetIcon({ size, ...rest }: IconProps) {
  return (
    <svg {...base(size)} {...rest}>
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}
