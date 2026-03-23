import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174, // 포트를 5174로 고정
    strictPort: true, // 5174가 사용 중일 때 자동으로 다른 포트로 넘어가지 않음
  },
});
