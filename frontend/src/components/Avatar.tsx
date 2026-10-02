import { avatarSrc } from '../lib/supabase'

const HUES = [262, 200, 160, 24, 330, 45, 290, 190]

export default function Avatar({ name, url, size = 28 }: { name: string; url?: string | null; size?: number }) {
  const src = avatarSrc(url)
  if (src) return <img className="avatar" src={src} alt="" width={size} height={size} />
  let h = 0
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  const hue = HUES[h % HUES.length]
  return (
    <span
      className="avatar"
      aria-hidden
      style={{ width: size, height: size, fontSize: size * 0.42, background: `oklch(0.62 0.13 ${hue})` }}
    >
      {name.slice(0, 1).toUpperCase()}
    </span>
  )
}
