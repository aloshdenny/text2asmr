import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Avatar from '../components/Avatar'
import Heatmap, { streaks } from '../components/Heatmap'
import { useAuth } from '../lib/auth'
import { supabase, timeZone } from '../lib/supabase'

type Stats = { username: string; display_name: string | null; avatar_url: string | null; joined: string; labelled: number; rank: number | null }

export default function Profile() {
  const { username = '' } = useParams()
  const { profile: me } = useAuth()
  const [stats, setStats] = useState<Stats | null | undefined>(undefined)
  const [counts, setCounts] = useState<Map<string, number>>(new Map())
  useEffect(() => {
    setStats(undefined)
    supabase.rpc('profile_stats', { p_username: username }).then(({ data }) => setStats((data as Stats[] | null)?.[0] ?? null))
    supabase.rpc('contributions', { p_username: username, p_tz: timeZone }).then(({ data }) =>
      setCounts(new Map(((data as { day: string; labelled: number }[] | null) ?? []).map((r) => [r.day, r.labelled]))),
    )
  }, [username])

  if (stats === undefined) return <main className="narrow"><p className="muted">Loading…</p></main>
  if (stats === null)
    return (
      <main className="narrow">
        <h1>No such listener</h1>
        <p className="muted">Nobody goes by @{username}. <Link to="/leaderboard">Back to the leaderboard</Link></p>
      </main>
    )
  const year = [...counts.values()].reduce((a, b) => a + b, 0)
  const s = streaks(counts)
  const best = Math.max(0, ...counts.values())
  return (
    <main className="profile">
      <div className="profile-head">
        <Avatar name={stats.username} url={stats.avatar_url} size={64} />
        <div>
          <h1>{stats.display_name || stats.username}</h1>
          <p className="muted">@{stats.username} · listening since {new Date(stats.joined).toLocaleDateString(undefined, { month: 'long', year: 'numeric' })}</p>
        </div>
        {me?.username === stats.username && <Link className="btn primary" to="/label">Label more</Link>}
      </div>
      <section className="stat-row">
        <div className="stat"><b>{stats.labelled.toLocaleString()}</b><span>clips labelled</span></div>
        <div className="stat"><b>{stats.rank ? `#${stats.rank}` : '—'}</b><span>all-time rank</span></div>
        <div className="stat"><b>{s.current}</b><span>day streak</span></div>
        <div className="stat"><b>{s.longest}</b><span>longest streak</span></div>
      </section>
      <section className="card">
        <div className="card-head">
          <h2>{year.toLocaleString()} clips in the last year</h2>
          {best > 0 && <span className="muted">best day: {best}</span>}
        </div>
        <Heatmap counts={counts} />
      </section>
    </main>
  )
}
