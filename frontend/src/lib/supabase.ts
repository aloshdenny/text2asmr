import { createClient } from '@supabase/supabase-js'

/** Supabase is reached only through this site's /sb proxy (vite.config.ts in dev, api/sb.ts on Vercel), which adds
 *  the API key server-side: no project URL or key ships in the bundle. "public" is a placeholder the proxy replaces. */
export const API = `${window.location.origin}/sb`

/** A random id for this browser, not tied to the person. It lets the site tell several accounts made on one device
 *  apart from several people; the /sb proxy scrambles it (with the network) before anything is stored. */
function deviceId(): string | null {
  try {
    let id = localStorage.getItem('asmrboard-device')
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem('asmrboard-device', id)
    }
    return id
  } catch {
    return null // storage blocked: no device mark, everything else works
  }
}

const device = deviceId()
export const supabase = createClient(API, 'public', device ? { global: { headers: { 'x-ab-device-id': device } } } : undefined)

/** How much of a clip has actually played, in ms (seeking past a part does not count it). */
export function playedMs(a: HTMLAudioElement): number {
  let t = 0
  for (let i = 0; i < a.played.length; i++) t += a.played.end(i) - a.played.start(i)
  return Math.round(t * 1000)
}

/** The whole clip has played once (within 0.3 s, for encoder padding); the database checks the same, a little looser.
 *  Uses the clip's stored length when the browser cannot tell how long the audio is. */
export function wholeClipPlayed(a: HTMLAudioElement, fallbackS: number): boolean {
  const d = Number.isFinite(a.duration) && a.duration > 0 ? a.duration : fallbackS
  return d > 0 && playedMs(a) >= d * 1000 - 300
}

/** An avatar is an https URL or an object in the public "avatars" bucket. */
export const avatarSrc = (url?: string | null) =>
  !url ? null : url.startsWith('avatars/') ? `${API}/storage/v1/object/public/${url}` : url

/** One label on the menu. `key` is what labels store; `title` (when set) is what the chip shows instead. */
export type Option = { key: string; grp: 'Voice' | 'Triggers' | 'NSFW' | 'Other'; hint: string; sort: number; title: string | null }

/** Keyboard shortcuts for the menu, in menu order. R is replay; U, freed when "can't tell" went, comes last. */
export const KEYS = '1234567890abcdefghijklmnopqstvwxyzu'

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
