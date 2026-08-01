/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Source Serif Pro"', "Georgia", "serif"],
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        ink: {
          DEFAULT: "#111111",
          muted: "#555555",
          subtle: "#888888",
        },
        paper: "#fafafa",
        rule: "#e5e5e5",
        // Accent: a calm teal that reads as "academic" rather than corporate.
        accent: {
          DEFAULT: "#0f766e",   // teal-700
          soft: "#ccfbf1",      // teal-100, used for hero wash + soft chips
          ink: "#134e4a",       // teal-900, used for darker accent text
        },
      },
    },
  },
  plugins: [],
};
