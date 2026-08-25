import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Docker Desktop bind mounts from a Windows host don't reliably deliver
    // inotify events into the container, so the default watcher silently
    // misses edits made from outside the container. Poll instead.
    watch: { usePolling: true },
  },
});
