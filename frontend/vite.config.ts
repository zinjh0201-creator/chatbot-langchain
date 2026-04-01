import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 5174, // 포트를 5174로 고정
//     strictPort: true, // 5174가 사용 중일 때 자동으로 다른 포트로 넘어가지 않음
//     allowedHosts: ["zin.tail2a3107.ts.net"],
//   },
// });

//tailscale 로 접속 가능하도록 수정
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174, // 포트를 5174로 고정
    strictPort: true, // 5174가 사용 중일 때 자동으로 다른 포트로 넘어가지 않음
    allowedHosts: ["zin.tail2a3107.ts.net"],
    host: true, // --host와 같은 역할
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
