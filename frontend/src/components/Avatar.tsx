import { avatarSrc } from '../lib/supabase'
import PixelCritter from './PixelCritter'
import { cx } from './ui'

/** A picture if there is one, else the person's own pixel critter (drawn from their username, the same every time). */
export default function Avatar({ name, url, size = 28, className }: { name: string; url?: string | null; size?: number; className?: string }) {
  const src = avatarSrc(url)
  const base = cx('inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full', className)
  if (src) return <img className={cx(base, 'object-cover')} src={src} alt="" width={size} height={size} style={{ width: size, height: size }} />
  return (
    <span aria-hidden className={base} style={{ width: size, height: size }}>
      <PixelCritter seed={name.toLowerCase()} size={size} />
    </span>
  )
}
