import { useEffect, useRef } from 'react'
import { dayKey } from '../lib/supabase'

const CELL = 11
const GAP = 3
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** GitHub-style contribution graph: the last 53 weeks, one column per week (Sunday on top). */
export default function Heatmap({ counts }: { counts: Map<string, number> }) {
  const wrap = useRef<HTMLDivElement>(null)
  // on narrow screens the graph scrolls: start at the newest week
  useEffect(() => {
    if (wrap.current) wrap.current.scrollLeft = wrap.current.scrollWidth
  }, [])
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const start = new Date(today)
  start.setDate(start.getDate() - (52 * 7 + today.getDay()))

  // four steps of the busiest day, so the biggest day is always the darkest cell
  const max = Math.max(0, ...counts.values())
  const level = (n: number) => (n === 0 || max === 0 ? 0 : Math.min(4, Math.ceil((4 * n) / max)))

  const cells: { x: number; y: number; key: string; n: number; date: Date }[] = []
  const months: { x: number; label: string }[] = []
  for (let d = new Date(start), i = 0; d <= today; d.setDate(d.getDate() + 1), i++) {
    const col = Math.floor(i / 7)
    const date = new Date(d)
    const key = dayKey(date)
    cells.push({ x: col, y: date.getDay(), key, n: counts.get(key) ?? 0, date })
    if (date.getDate() === 1 || i === 0) months.push({ x: col, label: MONTHS[date.getMonth()] })
  }
  const left = 28
  const top = 16
  const width = left + 53 * (CELL + GAP) - GAP // ends exactly at the last column, so it lines up with the legend
  const height = top + 7 * (CELL + GAP)
  return (
    <div>
      <div className="scroll-slim overflow-x-auto" ref={wrap}>
        <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto w-full min-w-[620px]" role="img" aria-label="Clips labelled per day over the last year">
          {months
            .filter((m, i) => i === 0 || m.x - months[i - 1].x >= 3)
            .map((m) => (
              <text
                key={`${m.x}-${m.label}`}
                // a month starting in the last columns is drawn right-aligned to the grid's edge instead of clipped
                x={left + m.x * (CELL + GAP) > width - 22 ? width : left + m.x * (CELL + GAP)}
                textAnchor={left + m.x * (CELL + GAP) > width - 22 ? 'end' : 'start'}
                y={10}
                className="fill-ink-muted text-[10px]"
              >
                {m.label}
              </text>
            ))}
          {[1, 3, 5].map((r) => (
            <text key={r} x={0} y={top + r * (CELL + GAP) + CELL - 2} className="fill-ink-muted text-[10px]">
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][r]}
            </text>
          ))}
          {cells.map((c) => (
            <rect key={c.key} x={left + c.x * (CELL + GAP)} y={top + c.y * (CELL + GAP)} width={CELL} height={CELL} rx={2.5} className={`hm-${level(c.n)}`}>
              <title>{`${c.n === 0 ? 'No' : c.n} clip${c.n === 1 ? '' : 's'} on ${c.date.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}`}</title>
            </rect>
          ))}
        </svg>
      </div>
      <div className="mt-2 flex items-center justify-end gap-1 text-xs text-ink-muted">
        Less
        {[0, 1, 2, 3, 4].map((l) => (
          <svg key={l} width={CELL} height={CELL} aria-hidden>
            <rect width={CELL} height={CELL} rx={2.5} className={`hm-${l}`} />
          </svg>
        ))}
        More
      </div>
    </div>
  )
}

/** Current streak (ending today, or yesterday if today is still empty) and longest streak, in days. */
// eslint-disable-next-line react-refresh/only-export-components
export function streaks(counts: Map<string, number>): { current: number; longest: number } {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  let longest = 0
  let run = 0
  const d = new Date(today)
  d.setDate(d.getDate() - 371)
  for (; d <= today; d.setDate(d.getDate() + 1)) {
    run = (counts.get(dayKey(d)) ?? 0) > 0 ? run + 1 : 0
    longest = Math.max(longest, run)
  }
  let current = 0
  const c = new Date(today)
  if ((counts.get(dayKey(c)) ?? 0) === 0) c.setDate(c.getDate() - 1)
  while ((counts.get(dayKey(c)) ?? 0) > 0) {
    current++
    c.setDate(c.getDate() - 1)
  }
  return { current, longest }
}
