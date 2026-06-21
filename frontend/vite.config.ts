import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Production assets are served from the nginx root.
export default defineConfig({
  base: "/",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
});
