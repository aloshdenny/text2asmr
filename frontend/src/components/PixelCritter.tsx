/* Pixel critters: everyone without a profile picture gets one, drawn in the browser from their username (the same name
   always gets the same critter). 16 x 16, 8-bit style: an animal head with headphones on, usually with its eyes
   closed because it is listening to something nice, a blush, and now and then a little "z". */

type Kind = 'cat' | 'bunny' | 'bear' | 'fox' | 'frog' | 'mouse' | 'owl'
const KINDS: Kind[] = ['cat', 'bunny', 'bear', 'fox', 'frog', 'mouse', 'owl']
const BODIES = ['#f6b48f', '#b9a3f5', '#8fd9b6', '#8cc8f5', '#f5d77f', '#f5a3c0', '#cfc9dc', '#c79a7b', '#9fe0e3', '#ffc98b']
const PHONES = ['#7c5cff', '#ff7a59', '#1fb5a3', '#3b3a4a', '#f2b400', '#ff5d8f', '#4c8dff']
const BGS = ['#efeaff', '#fff1e6', '#e6faf1', '#fdeef5', '#e8f4ff', '#fff8dc', '#f1f0f7']
const PINK = '#ff9fb4'
const INK = '#3b3a4a'
const N = 16

function hash(s: string) {
  let h = 2166136261
  for (const c of s) {
    h ^= c.codePointAt(0)!
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function shade(hex: string, f: number) {
  const n = parseInt(hex.slice(1), 16)
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.round(v * f))
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

export type CritterLook = { kind?: Kind; body?: string; phones?: string; bg?: string; sleepy?: boolean }
type Rect = [number, number, number, string] // x, y, width, colour (height 1)

function draw(seed: string, look: CritterLook): { bg: string; rects: Rect[] } {
  const r = rng(hash(seed))
  const pick = <T,>(xs: T[]) => xs[Math.floor(r() * xs.length)]
  const kind = look.kind ?? pick(KINDS)
  const body = look.body ?? (kind === 'frog' ? pick(['#8fd9b6', '#9fe0a0', '#b6e388']) : pick(BODIES))
  const phones = look.phones ?? pick(PHONES)
  const bg = look.bg ?? pick(BGS)
  const sleepy = look.sleepy ?? r() < 0.65
  const mouth = r() < 0.5 ? 'w' : 'smile'
  const zzz = sleepy && r() < 0.35

  const g: (string | null)[][] = Array.from({ length: N }, () => Array(N).fill(null))
  const set = (x: number, y: number, c: string) => { if (x >= 0 && x < N && y >= 0 && y < N) g[y][x] = c }
  const mirror = (x: number, y: number, c: string) => { set(x, y, c); set(N - 1 - x, y, c) }
  const disc = (cx: number, cy: number, rad: number) => {
    const out: [number, number][] = []
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= rad * rad) out.push([x, y])
    return out
  }

  // the shape: head plus ears, outlined as one piece
  const mask = Array.from({ length: N }, () => Array(N).fill(false))
  const fill = (x: number, y: number) => { if (x >= 0 && x < N && y >= 0 && y < N) mask[y][x] = true }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (((x + 0.5 - 8) / 5.6) ** 2 + ((y + 0.5 - 10) / 4.8) ** 2 <= 1) fill(x, y)
  const inner: [number, number][] = []
  const ear = (pts: [number, number][], ins: [number, number][] = []) => {
    for (const [x, y] of pts) { fill(x, y); fill(N - 1 - x, y) }
    for (const [x, y] of ins) inner.push([x, y], [N - 1 - x, y])
  }
  if (kind === 'cat') ear([[3, 3], [3, 4], [4, 4], [3, 5], [4, 5], [5, 5]], [[4, 5]])
  if (kind === 'fox') ear([[3, 2], [3, 3], [4, 3], [3, 4], [4, 4], [5, 4], [3, 5], [4, 5], [5, 5], [6, 5]], [[4, 4], [4, 5], [5, 5]])
  if (kind === 'bunny') ear([[5, 0], [6, 0], [5, 1], [6, 1], [5, 2], [6, 2], [5, 3], [6, 3], [5, 4], [6, 4], [5, 5], [6, 5]], [[6, 1], [6, 2], [6, 3]])
  if (kind === 'bear') ear(disc(4.5, 5.5, 1.8).filter(([x]) => x < 8), [[4, 5]])
  if (kind === 'mouse') ear(disc(4, 5, 2.5).filter(([x]) => x < 8), disc(4, 5, 1.2).filter(([x]) => x < 8))
  if (kind === 'frog') ear(disc(5, 6, 2).filter(([x]) => x < 8))
  if (kind === 'owl') ear([[3, 3], [3, 4], [4, 4], [4, 5]])

  // the headphone band arcs over the head, behind the ears
  for (let y = 0; y < 9; y++)
    for (let x = 0; x < N; x++) {
      const d = Math.hypot((x + 0.5 - 8) / 6.7, (y + 0.5 - 10) / 6.7)
      if (d >= 0.92 && d <= 1.08) set(x, y, shade(phones, 0.82))
    }
  const edge = shade(body, 0.68)
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (!mask[y][x]) continue
      const out = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => !mask[y + dy]?.[x + dx])
      set(x, y, out ? edge : body)
    }
  for (const [x, y] of inner) if (mask[y][x] && g[y][x] === body) set(x, y, PINK)

  // ear cups
  for (let y = 8; y <= 11; y++) { mirror(1, y, shade(phones, 0.75)); mirror(2, y, phones) }
  mirror(2, 7, shade(phones, 0.75)); mirror(2, 12, shade(phones, 0.75))

  // face
  if (kind === 'fox') for (const [x, y] of [[6, 12], [7, 12], [8, 12], [9, 12], [7, 13], [8, 13]] as [number, number][]) set(x, y, '#fff7ef')
  if (kind === 'owl') {
    for (const [x, y] of disc(5.5, 9.5, 1.6)) mirror(x, y, '#fffaf0')
    mirror(5, 9, INK)
    set(7, 11, '#f2a33a'); set(8, 11, '#f2a33a')
  } else if (kind === 'frog') {
    if (sleepy) { mirror(4, 6, INK); mirror(5, 6, INK) } else { mirror(5, 5, INK); mirror(5, 6, INK) }
  } else if (sleepy) {
    mirror(4, 10, INK); mirror(5, 10, INK)
  } else {
    mirror(5, 9, INK); mirror(5, 10, INK)
  }
  if (kind !== 'owl') {
    if (mouth === 'w') { set(6, 12, INK); set(7, 13, INK); set(8, 13, INK); set(9, 12, INK) }
    else { set(7, 12, INK); set(8, 12, INK); set(6, 11, INK); set(9, 11, INK) }
  }
  if (kind !== 'owl' && kind !== 'fox') mirror(4, 12, PINK)
  if (zzz) { set(12, 1, INK); set(13, 1, INK); set(13, 2, INK); set(12, 3, INK); set(13, 3, INK) }

  // rows of same-coloured pixels become one rect each
  const rects: Rect[] = []
  for (let y = 0; y < N; y++) {
    let x = 0
    while (x < N) {
      const c = g[y][x]
      if (!c) { x++; continue }
      let w = 1
      while (x + w < N && g[y][x + w] === c) w++
      rects.push([x, y, w, c])
      x += w
    }
  }
  return { bg, rects }
}

const cache = new Map<string, ReturnType<typeof draw>>()

export default function PixelCritter({ seed, size = 32, className, ...look }: { seed: string; size?: number; className?: string } & CritterLook) {
  const key = `${seed}|${look.kind ?? ''}|${look.body ?? ''}|${look.phones ?? ''}|${look.bg ?? ''}|${look.sleepy ?? ''}`
  let art = cache.get(key)
  if (!art) {
    art = draw(seed, look)
    cache.set(key, art)
  }
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} shapeRendering="crispEdges" aria-hidden="true" className={className}>
      <rect width="16" height="16" fill={art.bg} />
      {art.rects.map(([x, y, w, c], i) => <rect key={i} x={x} y={y} width={w} height={1} fill={c} />)}
    </svg>
  )
}
