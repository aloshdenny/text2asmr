import { useEffect, useState } from 'react'
import { cx } from './ui'

/** A progress ring with the percentage inside, like the battery widget: a quiet track and one coloured arc.
 *  `size` scales the whole counter (ring, stroke, number and captions) from the 136 px original. */
export default function Ring({ percent, color, label, caption, size = 136 }: { percent: number; color: string; label: string; caption?: string; size?: number }) {
  const k = size / 136
  const stroke = 11 * k
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const [shown, setShown] = useState(0) // starts empty and draws in
  useEffect(() => {
    const t = window.requestAnimationFrame(() => setShown(percent))
    return () => window.cancelAnimationFrame(t)
  }, [percent])
  return (
    <figure className={cx('flex flex-col items-center gap-3', k === 1 && 'w-[140px] sm:w-60')} style={k === 1 ? undefined : { width: '100%', maxWidth: size + 120 }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full -rotate-90" role="img" aria-label={`${percent.toFixed(1)}% ${label}${caption ? `, ${caption}` : ''}`}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-hover)" strokeWidth={stroke} />
          <circle
            className="ring-arc"
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={c * (1 - Math.min(100, Math.max(0, shown)) / 100)}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center font-black tracking-tight text-ink tabular-nums" style={{ fontSize: 28 * k }}>{percent.toFixed(1)}%</span>
      </div>
      <figcaption className="text-center">
        <span className="block font-extrabold text-ink" style={{ fontSize: 14 * k }}>{label}</span>
        {caption && <span className="block font-semibold text-ink-muted" style={{ fontSize: 12 * k }}>{caption}</span>}
      </figcaption>
    </figure>
  )
}
