import { avatarSrc } from '../lib/supabase'
import { cx } from './ui'

/** A picture if there is one, else the first letter on a neutral disc (no per-user colours). */
export default function Avatar({ name, url, size = 28, className }: { name: string; url?: string | null; size?: number; className?: string }) {
  const src = avatarSrc(url)
  const base = cx('inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full', className)
  if (src) return <img className={cx(base, 'object-cover')} src={src} alt="" width={size} height={size} style={{ width: size, height: size }} />
  return (
    <span
      aria-hidden
      className={cx(base, 'border border-hairline bg-surface-muted font-medium text-ink-secondary')}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.4) }}
    >
      {name.slice(0, 1).toUpperCase()}
    </span>
  )
}
