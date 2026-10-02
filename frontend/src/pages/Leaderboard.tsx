import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { AppLayout } from '../components/Layouts'
import { Card, EmptyState, Loading, PageHeader, cx } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useLeaderboard } from '../lib/stats'

const PERIODS: [string, number | null][] = [['All time', null], ['30 days', 30], ['7 days', 7]]

export default function Leaderboard() {
  const { profile } = useAuth()
  const [days, setDays] = useState<number | null>(null)
  const rows = useLeaderboard(days, 200)
  const mine = useRef<HTMLTableRowElement>(null)

  // signed in: bring your own row into view (only when it is off screen) and focus it. Waits a beat for layout and
  // reruns once the profile is known; if the smooth scroll gets cancelled or never starts (a background tab pauses
  // the animation), it jumps instead.
  const me = profile?.username
  useEffect(() => {
    const offscreen = (el: HTMLElement) => {
      const r = el.getBoundingClientRect()
      return r.top < 0 || r.bottom > window.innerHeight
    }
    let check = 0
    const t = window.setTimeout(() => {
      const el = mine.current
      if (!el) return
      el.focus({ preventScroll: true }) // before scrolling: focusing during a smooth scroll cancels it in Chrome
      if (!offscreen(el)) return
      el.scrollIntoView({ block: 'center', behavior: 'smooth' })
      check = window.setTimeout(() => offscreen(el) && el.scrollIntoView({ block: 'center' }), 700)
    }, 150)
    return () => {
      window.clearTimeout(t)
      window.clearTimeout(check)
    }
  }, [rows, me])

  return (
    <AppLayout>
      <PageHeader
        title="Leaderboard"
        description="Clips labelled, including the Ear Check kits from before the site."
        actions={
          <div role="tablist" className="inline-flex rounded-lg border border-hairline bg-surface-muted p-0.5">
            {PERIODS.map(([name, d]) => (
              <button
                key={name}
                role="tab"
                aria-selected={days === d}
                onClick={() => setDays(d)}
                className={cx(
                  'h-7 rounded-md px-3 text-[13px] font-medium transition-colors',
                  days === d ? 'bg-surface text-ink shadow-[0_1px_2px_rgb(0_0_0/0.08)]' : 'text-ink-muted hover:text-ink',
                )}
              >
                {name}
              </button>
            ))}
          </div>
        }
      />
      {rows === null ? (
        <Loading />
      ) : rows.length === 0 ? (
        <EmptyState title="Nobody has labelled a clip in this period yet." />
      ) : (
        <Card padded={false} className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-hairline text-left text-xs text-ink-muted">
                <th className="w-14 px-5 py-2.5 font-medium">#</th>
                <th className="w-full py-2.5 font-medium">Listener</th>
                <th className="whitespace-nowrap px-5 py-2.5 text-right font-medium">Clips</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {rows.map((r) => {
                const me = profile?.username === r.username
                return (
                  <tr
                    key={r.username}
                    ref={me ? mine : undefined}
                    tabIndex={me ? -1 : undefined}
                    aria-current={me ? 'true' : undefined}
                    className={cx(
                      'scroll-my-24 transition-colors outline-none',
                      me ? 'bg-brand-soft shadow-[inset_3px_0_0_var(--brand)]' : 'hover:bg-surface-hover',
                    )}
                  >
                    <td className={cx('px-5 py-3 font-mono text-xs', me ? 'text-brand' : 'text-ink-muted')}>{r.rank}</td>
                    <td className="py-3">
                      <Link to={`/u/${r.username}`} className="inline-flex items-center gap-2.5">
                        <Avatar name={r.username} url={r.avatar_url} size={26} />
                        <span className={cx('text-ink', me && 'font-semibold')}>{r.username}</span>
                        {me && <span className="text-ink-muted">(You)</span>}
                      </Link>
                    </td>
                    <td className="px-5 py-3 text-right font-mono tabular-nums text-ink">{r.labelled.toLocaleString()}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Card>
      )}
    </AppLayout>
  )
}
