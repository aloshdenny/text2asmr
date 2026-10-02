import { useEffect, useState } from 'react'
import { Button, cx } from './ui'

/** Headphones with sound waves breathing out of both ear cups. */
function HeadphonesArt() {
  return (
    <svg viewBox="0 0 120 96" className="hp-bob mx-auto h-24 w-28" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-brand">
        <path className="hp-wave hp-left" d="M19 54a12 12 0 0 0 0 16" />
        <path className="hp-wave hp-left d2" d="M12 49a19 19 0 0 0 0 26" />
        <path className="hp-wave hp-right" d="M101 54a12 12 0 0 1 0 16" />
        <path className="hp-wave hp-right d2" d="M108 49a19 19 0 0 1 0 26" />
      </g>
      <path d="M33 60V50a27 27 0 0 1 54 0v10" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" className="text-ink" />
      <rect x="25" y="52" width="15" height="24" rx="6" className="fill-ink" />
      <rect x="80" y="52" width="15" height="24" rx="6" className="fill-ink" />
      <rect x="29" y="57" width="7" height="14" rx="3.5" className="fill-brand" />
      <rect x="84" y="57" width="7" height="14" rx="3.5" className="fill-brand" />
    </svg>
  )
}

/** Shown when the labeller opens: asks for headphones, and its button is the click that lets the first clip play.
 *  onReady fires at once (start playback inside the click); the dialog then animates out and calls onClosed. */
export default function HeadphonesPrompt({ onReady, onClosed }: { onReady: () => void; onClosed: () => void }) {
  const [leaving, setLeaving] = useState(false)
  const ready = () => {
    if (leaving) return
    onReady()
    setLeaving(true)
    window.setTimeout(onClosed, 220)
  }
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === 'Escape' && ready()
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  })
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="hp-title"
      className={cx('hp-backdrop fixed inset-0 z-40 flex items-center justify-center bg-black/45 px-4 backdrop-blur-[3px]', leaving && 'hp-out')}
    >
      <div className="hp-panel w-full max-w-sm rounded-2xl border border-hairline bg-surface-raised px-6 pb-6 pt-7 text-center shadow-[0_28px_70px_-24px_rgb(0_0_0/0.6)]">
        <HeadphonesArt />
        <h2 id="hp-title" className="mt-4 text-lg font-semibold tracking-tight text-ink">Put your headphones on</h2>
        <p className="mt-2 text-sm leading-relaxed text-ink-secondary">
          These sounds are soft and recorded close to the mic. Headphones or earphones bring out the small details that
          tell one sound from another.
        </p>
        <Button variant="primary" size="lg" className="mt-6 w-full" onClick={ready} autoFocus>
          I’m ready
        </Button>
      </div>
    </div>
  )
}
