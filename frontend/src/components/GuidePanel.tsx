import { useEffect, useRef, useState } from 'react'
import { guideUrl, loadGuide, loadMenu, type GuideEntry, type Option } from '../lib/supabase'
import { Button, Kbd, cx } from './ui'

/** One example: a small play button with a progress ring; starting one example pauses any other. */
function GuideClip({ src, label }: { src: string; label: string }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [pct, setPct] = useState(0)
  const toggle = () => {
    const a = audio.current
    if (!a) return
    if (a.paused) {
      document.querySelectorAll('audio').forEach((o) => o !== a && o.pause())
      a.play().catch(() => {})
    } else a.pause()
  }
  const r = 13
  const c = 2 * Math.PI * r
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`${playing ? 'Pause' : 'Play'} ${label}`}
      className={cx(
        'inline-flex h-8 items-center gap-2 rounded-lg border pl-1 pr-2.5 text-xs font-medium transition-colors',
        playing ? 'border-brand-border bg-brand-soft text-brand' : 'border-hairline bg-surface text-ink-secondary hover:bg-surface-hover hover:text-ink',
      )}
    >
      <span className="relative flex h-6 w-6 items-center justify-center">
        <svg viewBox="0 0 30 30" className="absolute inset-0 h-6 w-6 -rotate-90" aria-hidden>
          <circle cx="15" cy="15" r={r} fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2" />
          <circle cx="15" cy="15" r={r} fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray={c} strokeDashoffset={c * (1 - pct)} strokeLinecap="round" />
        </svg>
        <svg viewBox="0 0 24 24" className="h-3 w-3 fill-current" aria-hidden>
          {playing ? (<><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></>) : <path d="M8 5.5v13l11-6.5z" />}
        </svg>
      </span>
      {label}
      <audio
        ref={audio}
        src={src}
        preload="none"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false)
          setPct(0)
        }}
        onTimeUpdate={(e) => setPct(e.currentTarget.duration ? e.currentTarget.currentTime / e.currentTarget.duration : 0)}
      />
    </button>
  )
}

/** Every label with what it covers, and example clips where the guide has them. */
export function GuideList() {
  const [menu, setMenu] = useState<Option[]>([])
  const [guide, setGuide] = useState<GuideEntry[]>([])
  useEffect(() => {
    loadMenu().then(setMenu, () => {})
    loadGuide().then(setGuide)
  }, [])
  const examples = new Map(guide.map((g) => [g.label, g.clips]))
  return (
    <div className="space-y-6">
      {(['Voice', 'Triggers', 'Other'] as const).map((g) => (
        <section key={g}>
          <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{g}</h3>
          <ul className="divide-y divide-hairline">
            {menu
              .filter((o) => o.grp === g)
              .map((o) => (
                <li key={o.key} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink">{o.key}</p>
                    <p className="text-sm text-ink-muted">{o.hint}</p>
                  </div>
                  <div className="flex shrink-0 gap-1.5">
                    {(examples.get(o.key) ?? []).map((c, i) => (
                      <GuideClip key={c} src={guideUrl(c)} label={`Example ${i + 1}`} />
                    ))}
                  </div>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

export default function GuidePanel({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-30 flex justify-end bg-black/30" onClick={onClose}>
      <aside className="scroll-slim h-full w-full max-w-md overflow-y-auto border-l border-hairline bg-surface-raised p-5" onClick={(e) => e.stopPropagation()} aria-label="Sound guide">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-ink">Sound guide</h2>
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close <Kbd>Esc</Kbd>
          </Button>
        </div>
        <GuideList />
      </aside>
    </div>
  )
}
