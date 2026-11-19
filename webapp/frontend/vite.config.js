import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: /api → the FastAPI backend (run_backend.sh, port 8790).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8790",
        changeOrigin: true,
      },
    },
  },
});
