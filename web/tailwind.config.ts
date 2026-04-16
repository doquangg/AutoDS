import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Subtle neutrals matching modern chat UIs
        canvas: "#fafaf9",
        ink: "#171717",
        muted: "#737373",
        accent: "#2563eb",
        success: "#16a34a",
        danger: "#dc2626",
        running: "#f59e0b",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
