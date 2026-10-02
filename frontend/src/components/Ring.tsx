import { useEffect, useState } from 'react'

/** A progress ring with the percentage inside, like the battery widget: a quiet track and one coloured arc. */
export default function Ring({ percent, color, label, caption }: { percent: number; color: string; label: string; caption?: string }) {
  const size = 136
  const stroke = 11
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const [shown, setShown] = useState(0) // starts empty and draws in
  useEffect(() => {
    const t = window.requestAnimationFrame(() => setShown(percent))
    return () => window.cancelAnimationFrame(t)
  }, [percent])
  return (
    <figure className="flex w-[136px] flex-col items-center gap-3 sm:w-56">
      <div className="relative" style={{ width: size, height: size }}>
        <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full -rotate-90" role="img" aria-label={`${percent}% ${label}`}>
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
        <span className="absolute inset-0 flex items-center justify-center text-[28px] font-semibold tracking-tight text-ink tabular-nums">{percent}%</span>
      </div>
      <figcaption className="text-center">
        <span className="block text-sm font-medium text-ink">{label}</span>
        {caption && <span className="block text-xs text-ink-muted">{caption}</span>}
      </figcaption>
    </figure>
  )
}
