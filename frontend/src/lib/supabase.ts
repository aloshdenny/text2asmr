import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined
const key = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined
if (!url || !key) throw new Error('Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY (see frontend/.env.example)')

export const supabase = createClient(url, key)
export const GITHUB_AUTH = import.meta.env.VITE_GITHUB_AUTH === '1'

/** Public files of the sound guide (bucket "guide"). */
export const guideUrl = (path: string) => `${url}/storage/v1/object/public/guide/${path}`

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
