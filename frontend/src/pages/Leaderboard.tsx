import { useState } from 'react'
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
  const rows = useLeaderboard(days, 100)
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
                <th className="hidden whitespace-nowrap px-5 py-2.5 text-right font-medium sm:table-cell">Last active</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {rows.map((r) => (
                <tr key={r.username} className={cx('transition-colors hover:bg-surface-hover', profile?.username === r.username && 'bg-brand-soft')}>
                  <td className="px-5 py-3 font-mono text-xs text-ink-muted">{r.rank}</td>
                  <td className="py-3">
                    <Link to={`/u/${r.username}`} className="inline-flex items-center gap-2.5">
                      <Avatar name={r.username} url={r.avatar_url} size={26} />
                      <span className="font-medium text-ink">{r.display_name || r.username}</span>
                      <span className="hidden text-ink-muted sm:inline">@{r.username}</span>
                    </Link>
                  </td>
                  <td className="px-5 py-3 text-right font-mono tabular-nums text-ink">{r.labelled.toLocaleString()}</td>
                  <td className="hidden px-5 py-3 text-right text-ink-muted sm:table-cell">{new Date(r.last_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </AppLayout>
  )
}
