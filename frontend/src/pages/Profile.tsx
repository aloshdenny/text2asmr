import { useParams } from 'react-router-dom'
import Avatar from '../components/Avatar'
import Heatmap, { streaks } from '../components/Heatmap'
import { AppLayout } from '../components/Layouts'
import { ButtonLink, Card, CardHeader, EmptyState, Loading, PageHeader, Stat } from '../components/ui'
import { useAuth } from '../lib/auth'
import { since, useUserStats } from '../lib/stats'

/** A listener's public page. */
export default function Profile() {
  const { username = '' } = useParams()
  const { profile: me } = useAuth()
  const { stats, counts } = useUserStats(username)
  if (stats === undefined) return <AppLayout><Loading /></AppLayout>
  if (stats === null)
    return (
      <AppLayout>
        <EmptyState title="No such labeller" description={`Nobody goes by @${username}.`} action={<ButtonLink to="/leaderboard">Back to the leaderboard</ButtonLink>} />
      </AppLayout>
    )
  const s = streaks(counts)
  const year = [...counts.values()].reduce((a, b) => a + b, 0)
  const best = Math.max(0, ...counts.values())
  const mine = me?.username === stats.username
  return (
    <AppLayout>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            <Avatar name={stats.username} url={stats.avatar_url} size={40} />
            <span>
              {stats.display_name || stats.username}
              <span className="block text-sm font-normal text-ink-muted">@{stats.username} · labelling since {since(stats.joined)}</span>
            </span>
          </span>
        }
        actions={mine ? <ButtonLink to="/settings" variant="ghost">Edit profile</ButtonLink> : undefined}
      />
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value={stats.labelled.toLocaleString()} label="clips labelled" />
        <Stat value={stats.rank ? `#${stats.rank}` : 'Unranked'} label="overall rank" />
        <Stat value={`${s.current} ${s.current === 1 ? 'day' : 'days'}`} label="current streak" />
        <Stat value={best} label="best day" />
      </section>
      <Card className="mt-3">
        <CardHeader title={`${year.toLocaleString()} clips in the last year`} />
        <Heatmap counts={counts} />
      </Card>
    </AppLayout>
  )
}
