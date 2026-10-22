/* Referral links (/r/<handle>, one per creator). The handle waits in this browser until the visitor has a profile,
   then set_referral credits it once (the database ignores it for anyone who was already labelling). */

const KEY = 'asmrboard:ref'
const TTL = 30 * 24 * 3600 * 1000
const HANDLE = /^[a-z0-9._]{1,30}$/

export function saveReferral(handle: string) {
  const h = handle.trim().toLowerCase()
  if (!HANDLE.test(h)) return
  try {
    if (!localStorage.getItem(KEY)) localStorage.setItem(KEY, JSON.stringify({ h, at: Date.now() })) // first link wins
  } catch { /* storage blocked: no credit */ }
}

export function pendingReferral(): string | null {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null') as { h: string; at: number } | null
    if (v && HANDLE.test(v.h) && Date.now() - v.at < TTL) return v.h
    if (v) localStorage.removeItem(KEY)
  } catch { /* ignore */ }
  return null
}

export function clearReferral() {
  try { localStorage.removeItem(KEY) } catch { /* ignore */ }
}
