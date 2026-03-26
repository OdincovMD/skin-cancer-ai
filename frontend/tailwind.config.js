/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        med: {
          50:  "#f0fdfa",
          100: "#ccfbf1",
          200: "#99f6e4",
          300: "#5eead4",
          400: "#2dd4bf",
          500: "#14b8a6",
          600: "#0d9488",
          700: "#0f766e",
          800: "#115e59",
          900: "#134e4a",
          950: "#042f2e",
        },
      },
      keyframes: {
        slideIn: {
          "0%":   { opacity: 0, transform: "translateY(4px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        fadeIn: {
          "0%":   { opacity: 0 },
          "100%": { opacity: 1 },
        },
        pulse: {
          "0%, 100%": { opacity: 1 },
          "50%":      { opacity: 0.5 },
        },
        spin: {
          from: { transform: "rotate(0deg)" },
          to:   { transform: "rotate(360deg)" },
        },
      },
      animation: {
        slideIn:  "slideIn .2s ease-out forwards var(--delay, 0s)",
        fadeIn:   "fadeIn .3s ease-out forwards",
        pulse:    "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        spin:     "spin 1s linear infinite",
      },
    },
  },
  plugins: [],
}
