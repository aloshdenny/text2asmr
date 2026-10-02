import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import Heatmap, { streaks } from '../components/Heatmap'
import { AppLayout } from '../components/Layouts'
import { ButtonLink, Card, CardHeader, Loading, PageHeader, Stat } from '../components/ui'
import { useAuth } from '../lib/auth'
import { since, useLeaderboard, useUserStats } from '../lib/stats'
import { dayKey } from '../lib/supabase'

export default function Dashboard() {
  const { profile } = useAuth()
  const { stats, counts } = useUserStats(profile?.username)
  const top = useLeaderboard(30, 5)
  if (!profile || stats === undefined) return <AppLayout><Loading /></AppLayout>
  const s = streaks(counts)
  const today = counts.get(dayKey(new Date())) ?? 0
  const year = [...counts.values()].reduce((a, b) => a + b, 0)
  return (
    <AppLayout>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            <Avatar name={profile.username} url={profile.avatar_url} size={40} />
            <span>
              {profile.display_name || profile.username}
              <span className="block text-sm font-normal text-ink-muted">
                @{profile.username}
                {stats ? ` · listening since ${since(stats.joined)}` : ''}
              </span>
            </span>
          </span>
        }
        actions={
          <>
            <ButtonLink to={`/u/${profile.username}`} variant="ghost">Public profile</ButtonLink>
            <ButtonLink to="/label" variant="primary">Start labelling</ButtonLink>
          </>
        }
      />

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value={(stats?.labelled ?? 0).toLocaleString()} label="clips labelled" />
        <Stat value={stats?.rank ? `#${stats.rank}` : '—'} label="all-time rank" />
        <Stat value={today} label="today" />
        <Stat value={`${s.current} ${s.current === 1 ? 'day' : 'days'}`} label={`streak · best ${s.longest}`} />
      </section>

      <Card className="mt-3">
        <CardHeader title={`${year.toLocaleString()} clips in the last year`} />
        <Heatmap counts={counts} />
      </Card>

      <section className="mt-3 grid gap-3 md:grid-cols-2">
        <Card>
          <CardHeader title="Top listeners" description="Last 30 days" action={<Link to="/leaderboard" className="text-sm text-brand hover:underline">Leaderboard</Link>} />
          <ol className="divide-y divide-hairline">
            {(top ?? []).map((r) => (
              <li key={r.username} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                <span className="w-4 text-right font-mono text-xs text-ink-muted">{r.rank}</span>
                <Avatar name={r.username} url={r.avatar_url} size={24} />
                <Link to={`/u/${r.username}`} className="min-w-0 flex-1 truncate text-sm text-ink hover:underline">
                  {r.display_name || r.username}
                  {r.username === profile.username && <span className="ml-1.5 text-xs text-ink-muted">you</span>}
                </Link>
                <span className="font-mono text-sm tabular-nums text-ink-secondary">{r.labelled.toLocaleString()}</span>
              </li>
            ))}
          </ol>
        </Card>
        <Card>
          <CardHeader title="Sound guide" description="What each label covers, with example clips." />
          <p className="text-sm text-ink-secondary">
            Listeners who mean the same thing by “crinkling” or “tapping” make every label worth more. Two minutes with
            the guide is the best thing you can do for your accuracy.
          </p>
          <div className="mt-4 flex gap-2">
            <ButtonLink to="/guide" size="sm">Open the guide</ButtonLink>
            <ButtonLink to="/settings" size="sm" variant="ghost">Edit profile</ButtonLink>
          </div>
        </Card>
      </section>
    </AppLayout>
  )
}
