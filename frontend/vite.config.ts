import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev mirrors the Vercel proxy (api/sb.ts): /sb/* goes to Supabase with the key added here, never in the bundle.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const key = env.SUPABASE_ANON_KEY ?? ''
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/sb': {
          target: env.SUPABASE_URL || 'http://127.0.0.1:54321',
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/sb/, ''),
          configure: (proxy) =>
            proxy.on('proxyReq', (req) => {
              req.setHeader('apikey', key)
              const auth = req.getHeader('authorization')
              if (!auth || auth === 'Bearer public') req.setHeader('authorization', `Bearer ${key}`)
            }),
        },
      },
    },
  }
})
