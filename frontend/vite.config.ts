import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
      },
    },
  },
  build: {
    target: 'esnext',
    // Default Vite minifier (esbuild). `terser` was failing the build
    // because it is an optional peer dep and was not in package.json.
    minify: 'esbuild',
  },
});

