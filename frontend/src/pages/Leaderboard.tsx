import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'

type Row = { rank: number; username: string; display_name: string | null; avatar_url: string | null; labelled: number; last_at: string }
const PERIODS: [string, number | null][] = [['All time', null], ['30 days', 30], ['7 days', 7]]

export default function Leaderboard() {
  const { profile } = useAuth()
  const [days, setDays] = useState<number | null>(null)
  const [rows, setRows] = useState<Row[] | null>(null)
  useEffect(() => {
    setRows(null)
    supabase.rpc('leaderboard', { p_days: days, p_limit: 100 }).then(({ data }) => setRows((data as Row[] | null) ?? []))
  }, [days])
  return (
    <main className="narrow">
      <div className="page-head">
        <h1>Leaderboard</h1>
        <div className="tabs" role="tablist">
          {PERIODS.map(([name, d]) => (
            <button key={name} role="tab" aria-selected={days === d} className={days === d ? 'on' : ''} onClick={() => setDays(d)}>
              {name}
            </button>
          ))}
        </div>
      </div>
      <div className="card board">
        {rows === null ? (
          <p className="muted">Loading…</p>
        ) : rows.length === 0 ? (
          <p className="muted">Nobody has labelled a clip in this period yet.</p>
        ) : (
          <table>
            <thead>
              <tr><th>#</th><th>Listener</th><th className="num">Clips</th><th className="num hide-sm">Last active</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.username} className={profile?.username === r.username ? 'me' : ''}>
                  <td className={`rank r${r.rank}`}>{r.rank}</td>
                  <td>
                    <Link to={`/u/${r.username}`} className="who">
                      <Avatar name={r.username} url={r.avatar_url} size={26} />
                      <span>{r.display_name || r.username}</span>
                      {r.display_name && <span className="muted">@{r.username}</span>}
                    </Link>
                  </td>
                  <td className="num"><b>{r.labelled.toLocaleString()}</b></td>
                  <td className="num muted hide-sm">{new Date(r.last_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </main>
  )
}
