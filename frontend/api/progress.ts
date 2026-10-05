// The home page's ring numbers, cached on Vercel's CDN: dataset_progress() serves a once-a-day snapshot, so nearly
// every visitor gets this from the edge without touching the database. Only the four counts and the snapshot time
// leave here; the key is added server-side, as in api/sb.ts.
export const config = { runtime: 'edge' }

declare const process: { env: Record<string, string | undefined> }

export default async function handler(): Promise<Response> {
  const base = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_ANON_KEY
  if (!base || !key) return new Response('not configured', { status: 503 })
  const res = await fetch(`${base}/rest/v1/rpc/dataset_progress`, {
    method: 'POST',
    headers: { apikey: key, authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: '{}',
  })
  if (!res.ok) return new Response('unavailable', { status: 502, headers: { 'cache-control': 'no-store' } })
  const [row] = (await res.json()) as Record<string, unknown>[]
  if (!row) return new Response('unavailable', { status: 502, headers: { 'cache-control': 'no-store' } })
  const { foundation_items, foundation_human_labelled, corpus_recordings, ai_labelled_recordings, computed_at } = row
  return Response.json(
    { foundation_items, foundation_human_labelled, corpus_recordings, ai_labelled_recordings, computed_at },
    {
      headers: {
        'cache-control': 'public, max-age=300', // browsers: 5 minutes
        'vercel-cdn-cache-control': 'max-age=3600, stale-while-revalidate=86400', // edge: an hour, stale while it refetches
      },
    },
  )
}
