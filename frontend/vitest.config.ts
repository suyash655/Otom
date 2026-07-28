import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./vitest.setup.ts",
    include: ["**/*.{test,spec}.{js,mjs,cjs,ts,mts,jsx,tsx}"],
    coverage: {
      reporter: ["text", "lcov"],
      exclude: ["**/node_modules/**", "**/.next/**", "**/coverage/**"],
    },
  },
});
