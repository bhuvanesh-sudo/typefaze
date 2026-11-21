/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.TYPEFAZE_BASE ?? "/",
  plugins: [react()],
  // The model worker loads ONNX Runtime on demand, which needs an ES-module worker.
  worker: { format: "es" },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
