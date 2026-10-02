import { createClient } from '@supabase/supabase-js'

/** Supabase is reached only through this site's /sb proxy (vite.config.ts in dev, api/sb.ts on Vercel), which adds
 *  the API key server-side: no project URL or key ships in the bundle. "public" is a placeholder the proxy replaces. */
export const API = `${window.location.origin}/sb`
export const supabase = createClient(API, 'public')

/** Public files of the sound guide (bucket "guide"). */
export const guideUrl = (path: string) => `${API}/storage/v1/object/public/guide/${path}`

/** An avatar is an https URL or an object in the public "avatars" bucket. */
export const avatarSrc = (url?: string | null) =>
  !url ? null : url.startsWith('avatars/') ? `${API}/storage/v1/object/public/${url}` : url

export type Option = { key: string; grp: 'Voice' | 'Triggers' | 'Other'; hint: string; sort: number }
export type GuideEntry = { label: string; hint: string; clips: string[] }

/** Shortcut keys in menu order: 1-0 for the first ten labels, then a-j. */
export const KEYS = '1234567890abcdefghij'

export async function loadMenu(): Promise<Option[]> {
  const { data, error } = await supabase.from('label_options').select('key, grp, hint, sort').order('sort')
  if (error) throw error
  return data as Option[]
}

export async function loadGuide(): Promise<GuideEntry[]> {
  try {
    const r = await fetch(guideUrl('guide.json'), { cache: 'no-cache' })
    return r.ok ? ((await r.json()) as GuideEntry[]) : []
  } catch {
    return []
  }
}

/** Local calendar day as YYYY-MM-DD (matches contributions() days in the viewer's time zone). */
export function dayKey(d: Date): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

export const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'

/** Errors people can act on: a server or network failure reads as one, not as "HTTP 503" or "Failed to fetch". */
export function friendlyError(message: string): string {
  if (/^HTTP 5\d\d$|failed to fetch|networkerror|load failed|not configured/i.test(message))
    return 'Can’t reach ASMR Board right now. Please try again in a moment.'
  return message
}
