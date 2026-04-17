import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#faf7f0",
        canvasDeep: "#f4efe2",
        surface: "#ffffff",
        sidebar: "#f3edde",
        ink: "#2a2622",
        inkSoft: "#413b35",
        muted: "#7a736a",
        subtle: "#a8a097",
        border: "#eae3d2",
        borderStrong: "#d6ccb4",
        accent: "#c96442",
        accentHover: "#b05537",
        accentSoft: "#f3e3d6",
        success: "#3f8a4d",
        danger: "#c0392b",
        running: "#c7894a",
        userBubble: "#efe7d2",
      },
      fontFamily: {
        sans: [
          "Geist",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        display: [
          "Instrument Serif",
          "ui-serif",
          "Georgia",
          "serif",
        ],
        mono: [
          "Geist Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        card: "0 1px 2px 0 rgb(42 38 34 / 0.04)",
        cardHover: "0 6px 20px -8px rgb(42 38 34 / 0.14)",
        composer:
          "0 2px 6px -2px rgb(42 38 34 / 0.06), 0 18px 44px -16px rgb(42 38 34 / 0.14)",
        pill: "0 1px 3px 0 rgb(42 38 34 / 0.06), 0 1px 2px -1px rgb(42 38 34 / 0.06)",
        soft: "0 1px 2px 0 rgb(42 38 34 / 0.03), 0 1px 3px 0 rgb(42 38 34 / 0.04)",
      },
      borderRadius: {
        "4xl": "2rem",
      },
      animation: {
        "spin-slow": "spin-slow 1.4s linear infinite",
        "pulse-dot": "pulse-dot 1.5s ease-in-out infinite",
        "fade-in": "fade-in 260ms ease-out",
        "slide-up": "slide-up 320ms cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-up-lg": "slide-up-lg 520ms cubic-bezier(0.16, 1, 0.3, 1)",
        shimmer: "shimmer 2.4s ease-in-out infinite",
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
        "slide-up-lg": {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
