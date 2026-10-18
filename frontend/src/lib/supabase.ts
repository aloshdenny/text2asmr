import { createClient } from '@supabase/supabase-js'

/** Supabase is reached only through this site's /sb proxy (vite.config.ts in dev, api/sb.ts on Vercel), which adds
 *  the API key server-side: no project URL or key ships in the bundle. "public" is a placeholder the proxy replaces. */
export const API = `${window.location.origin}/sb`
export const supabase = createClient(API, 'public')

/** An avatar is an https URL or an object in the public "avatars" bucket. */
export const avatarSrc = (url?: string | null) =>
  !url ? null : url.startsWith('avatars/') ? `${API}/storage/v1/object/public/${url}` : url

/** One label on the menu. `key` is what labels store; `title` (when set) is what the chip shows instead. */
export type Option = { key: string; grp: 'Voice' | 'Triggers' | 'NSFW' | 'Other'; hint: string; sort: number; title: string | null }

/** Shortcut keys in menu order: 1-0 for the first ten labels, then a-j. */
/** Keyboard shortcuts for the menu, in menu order. R (replay) and U (can't tell) are taken. */
export const KEYS = '1234567890abcdefghijklmnopqstvwxyz'

export async function loadMenu(): Promise<Option[]> {
  const { data, error } = await supabase.from('label_options').select('key, grp, hint, sort, title').order('sort')
  if (error) throw error
  return data as Option[]
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
