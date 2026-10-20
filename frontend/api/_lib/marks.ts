// Scrambled marks of where a request comes from, so the site can tell several accounts made on one device (or one
// network) apart from several people, without ever storing who anyone is. The proxy (api/sb.ts) computes them and
// passes them to Supabase as x-ab-* headers; the database keeps only these values (private.user_marks).
//
//   x-ab-device  keyed hash of the random id this browser keeps in localStorage (asmrboard-device)
//   x-ab-ip      keyed hash of the client IP
//   x-ab-net     keyed hash of its network (IPv4 /24, IPv6 /48): one household, office or mobile cell
//   x-ab-ua      coarse browser and OS ("Chrome Android"), in clear: shared by millions, identifies no one
//
// The hashes are HMAC-SHA256 under TRACK_SALT, a secret that exists only in Vercel's environment: without it a hash
// cannot be turned back into an IP, not even by trying every IPv4 address.

const enc = new TextEncoder()
let cached: { salt: string; key: Promise<CryptoKey> } | null = null

function hmacKey(salt: string): Promise<CryptoKey> {
  if (!cached || cached.salt !== salt) {
    cached = { salt, key: crypto.subtle.importKey('raw', enc.encode(salt), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']) }
  }
  return cached.key
}

async function scramble(salt: string, kind: string, value: string): Promise<string> {
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', await hmacKey(salt), enc.encode(`${kind}:${value}`)))
  let s = ''
  for (const b of sig) s += String.fromCharCode(b)
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '').slice(0, 22)
}

export function clientIp(h: Headers): string | null {
  const ip = (h.get('x-real-ip') ?? h.get('x-forwarded-for')?.split(',')[0] ?? '').trim()
  return ip || null
}

/** The /24 of an IPv4 address or the /48 of an IPv6 one. */
export function network(ip: string): string {
  if (ip.includes('.') && !ip.includes(':')) return ip.split('.').slice(0, 3).join('.') + '.0/24'
  const [head, tail = ''] = ip.split('::')
  const a = head ? head.split(':') : []
  const b = tail ? tail.split(':') : []
  const full = ip.includes('::') ? [...a, ...Array(8 - a.length - b.length).fill('0'), ...b] : a
  return full.slice(0, 3).map((x) => x.toLowerCase().replace(/^0+(?=.)/, '')).join(':') + '::/48'
}

export function uaFamily(ua: string): string {
  const os = /Android/.test(ua) ? 'Android' : /iPhone|iPad|iPod/.test(ua) ? 'iOS' : /Windows/.test(ua) ? 'Windows'
    : /Mac OS X|Macintosh/.test(ua) ? 'macOS' : /CrOS/.test(ua) ? 'ChromeOS' : /Linux/.test(ua) ? 'Linux' : 'Other'
  const browser = /Edg\//.test(ua) ? 'Edge' : /OPR\/|Opera/.test(ua) ? 'Opera' : /Firefox\//.test(ua) ? 'Firefox'
    : /SamsungBrowser/.test(ua) ? 'Samsung' : /Chrome\//.test(ua) ? 'Chrome' : /Safari\//.test(ua) ? 'Safari' : 'Other'
  return `${browser} ${os}`
}

/** The x-ab-* headers for a request (whatever can be worked out from it). */
export async function marks(h: Headers, salt: string): Promise<Record<string, string>> {
  const out: Record<string, string> = {}
  const device = h.get('x-ab-device-id')
  if (device && /^[A-Za-z0-9-]{8,64}$/.test(device)) out['x-ab-device'] = await scramble(salt, 'device', device)
  const ip = clientIp(h)
  if (ip) {
    out['x-ab-ip'] = await scramble(salt, 'ip', ip)
    out['x-ab-net'] = await scramble(salt, 'net', network(ip))
  }
  const ua = h.get('user-agent')
  if (ua) out['x-ab-ua'] = uaFamily(ua)
  return out
}
