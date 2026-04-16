import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f7f7f5",
        surface: "#ffffff",
        ink: "#18181b",
        muted: "#71717a",
        subtle: "#a1a1aa",
        border: "#e4e4e7",
        borderStrong: "#d4d4d8",
        accent: "#2563eb",
        accentHover: "#1d4ed8",
        success: "#16a34a",
        danger: "#dc2626",
        running: "#f59e0b",
        userBubble: "#2563eb",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        card: "0 1px 2px 0 rgb(0 0 0 / 0.04), 0 1px 3px 0 rgb(0 0 0 / 0.05)",
        cardHover: "0 4px 10px -2px rgb(0 0 0 / 0.08)",
        composer: "0 -2px 20px -8px rgb(0 0 0 / 0.08)",
      },
      animation: {
        "spin-slow": "spin-slow 1.4s linear infinite",
        "pulse-dot": "pulse-dot 1.5s ease-in-out infinite",
        "fade-in": "fade-in 240ms ease-out",
        "slide-up": "slide-up 280ms cubic-bezier(0.16, 1, 0.3, 1)",
      },
      keyframes: {
        "spin-slow": {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
        "pulse-dot": {
          "0%, 100%": { opacity: "0.4", transform: "scale(0.85)" },
          "50%": { opacity: "1", transform: "scale(1)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
