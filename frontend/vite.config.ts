import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by FastAPI under /static (see api/main.py).
export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
});
