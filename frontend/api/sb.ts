// Same-origin proxy to Supabase (Vercel Edge Function): the browser only ever talks to <site>/sb/..., so neither the
// project URL nor its API key is in the bundle. The key is added here from server-side env vars.
// Only the auth, REST and storage APIs pass; everything the browser may do is still decided by row-level security.
export const config = { runtime: 'edge' }

declare const process: { env: Record<string, string | undefined> }

const ALLOWED = /^(auth|rest|storage)\/v1\//

export default async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url)
  const path = url.searchParams.get('__p') ?? ''
  url.searchParams.delete('__p')
  url.searchParams.delete('sbpath') // Vercel also passes the rewrite's route segment as a query parameter
  if (!ALLOWED.test(path)) return new Response('not found', { status: 404 })
  const base = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_ANON_KEY
  if (!base || !key) return new Response('not configured', { status: 503 })

  const headers = new Headers()
  for (const [k, v] of req.headers) if (!/^(host|connection|content-length|x-forwarded-|x-vercel-|x-real-ip|forwarded|cookie)/i.test(k)) headers.set(k, v)
  headers.set('apikey', key)
  const auth = headers.get('authorization')
  if (!auth || auth === 'Bearer public') headers.set('authorization', `Bearer ${key}`)

  const body = req.method === 'GET' || req.method === 'HEAD' ? undefined : await req.arrayBuffer()
  const res = await fetch(`${base}/${path}${url.search}`, { method: req.method, headers, body, redirect: 'manual' })
  const out = new Headers(res.headers)
  out.delete('content-encoding') // fetch already decoded the body
  out.delete('content-length')
  const loc = out.get('location')
  if (loc?.startsWith(base)) out.set('location', `${url.origin}/sb${loc.slice(base.length)}`)
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: out })
}
