import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import Heatmap, { streaks } from '../components/Heatmap'
import { AppLayout } from '../components/Layouts'
import { ButtonLink, Card, CardHeader, Loading, PageHeader, Stat } from '../components/ui'
import { useAuth } from '../lib/auth'
import { since, useUserStats } from '../lib/stats'
import { dayKey } from '../lib/supabase'

export function ArenaPill() {
  return (
    <span className="hum inline-flex">
      <Link to="/leaderboard" className="inline-flex h-8 items-center gap-1.5 px-3.5 text-[13px] font-medium text-ink transition-colors hover:text-brand">
        Check out the Arena <span aria-hidden>✨</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="h-3.5 w-3.5">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      </Link>
    </span>
  )
}

export default function Dashboard() {
  const { profile } = useAuth()
  const { stats, counts } = useUserStats(profile?.username)
  if (!profile || stats === undefined) return <AppLayout><Loading /></AppLayout>
  const s = streaks(counts)
  const today = counts.get(dayKey(new Date())) ?? 0
  const year = [...counts.values()].reduce((a, b) => a + b, 0)
  return (
    <AppLayout>
      <div className="mb-6">
        <ArenaPill />
      </div>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            <Avatar name={profile.username} url={profile.avatar_url} size={40} />
            <span>
              {profile.display_name || profile.username}
              <span className="block text-sm font-normal text-ink-muted">
                @{profile.username}
                {stats ? ` · labelling since ${since(stats.joined)}` : ''}
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
        <Stat value={stats?.rank ? `#${stats.rank}` : 'Unranked'} label="overall rank" />
        <Stat value={today} label="today" />
        <Stat value={`${s.current} ${s.current === 1 ? 'day' : 'days'}`} label={`streak · best ${s.longest}`} />
      </section>

      <Card className="mt-3">
        <CardHeader title={`${year.toLocaleString()} clips in the last year`} />
        <Heatmap counts={counts} />
      </Card>
    </AppLayout>
  )
}
