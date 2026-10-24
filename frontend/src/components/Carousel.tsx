import { useRef } from 'react'
import { Kbd, cx } from './ui'
import type { Option } from '../lib/supabase'

/** One category's labels in a row that scrolls sideways, with arrows. Used by both labellers: `count` says how many
 *  times a label is already picked (a tick is 1, a timeline label can have several spans). */
export default function Carousel({ title, options, count, onPick, full = false, keyOf, onHint }: {
  title: string
  options: Option[]
  count: (k: string) => number
  onPick: (o: Option) => void
  full?: boolean
  keyOf?: (o: Option) => string
  onHint?: (o: Option) => void
}) {
  const row = useRef<HTMLDivElement>(null)
  const by = (dir: number) => row.current?.scrollBy({ left: dir * row.current.clientWidth * 0.8, behavior: 'smooth' })
  return (
    <div>
      <h3 className="mb-2 text-xs font-extrabold uppercase tracking-[0.08em] text-ink-muted">{title}</h3>
      <div className="flex items-center gap-1.5">
        <button type="button" aria-label={`Earlier ${title} labels`} onClick={() => by(-1)} className="h-10 w-8 shrink-0 rounded-xl text-lg font-black text-ink-muted hover:bg-surface-hover hover:text-ink">‹</button>
        <div ref={row} className="no-scrollbar flex min-w-0 flex-1 snap-x gap-1.5 overflow-x-auto scroll-smooth">
          {options.map((o) => {
            const n = count(o.key)
            return (
              <button key={o.key} type="button" title={o.hint} aria-pressed={n > 0} disabled={full && n === 0}
                onMouseEnter={() => onHint?.(o)} onFocus={() => onHint?.(o)} onClick={() => onPick(o)}
                className={cx('press mb-1 inline-flex h-10 shrink-0 snap-start items-center gap-1.5 rounded-2xl border-2 px-3 text-sm font-bold [--lift:3px] disabled:opacity-40',
                  n ? 'pop border-brand-border bg-brand-soft text-brand [--edge:var(--brand-border)]' : 'border-hairline bg-surface text-ink [--edge:var(--border)] hover:bg-surface-hover')}>
                {keyOf && <Kbd className={cx('max-sm:hidden', n > 0 && 'border-brand-border bg-transparent text-brand')}>{keyOf(o)}</Kbd>}
                {o.title ?? o.key}
                {n > 1 && <span className="font-mono text-[11px]">×{n}</span>}
              </button>
            )
          })}
        </div>
        <button type="button" aria-label={`More ${title} labels`} onClick={() => by(1)} className="h-10 w-8 shrink-0 rounded-xl text-lg font-black text-ink-muted hover:bg-surface-hover hover:text-ink">›</button>
      </div>
    </div>
  )
}
