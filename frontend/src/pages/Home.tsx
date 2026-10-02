import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'

type Stats = { labels: number; labellers: number; clips: number; clips_complete: number }
type Row = { rank: number; username: string; display_name: string | null; avatar_url: string | null; labelled: number }

export default function Home() {
  const { session } = useAuth()
  const [stats, setStats] = useState<Stats | null>(null)
  const [top, setTop] = useState<Row[]>([])
  useEffect(() => {
    supabase.rpc('site_stats').then(({ data }) => setStats((data as Stats[] | null)?.[0] ?? null))
    supabase.rpc('leaderboard', { p_days: 30, p_limit: 5 }).then(({ data }) => setTop((data as Row[] | null) ?? []))
  }, [])
  const fmt = (n?: number) => (n ?? 0).toLocaleString()
  return (
    <main className="home">
      <section className="hero">
        <p className="eyebrow">A community dataset for text2asmr</p>
        <h1>Teach machines to hear ASMR.</h1>
        <p className="lede">
          Listen to six-second clips and tick the sounds you hear — whispers, tapping, crinkles, brushing. Every clip
          goes to several listeners independently, and your answers train the open ASMR sound classifier behind text2asmr.
        </p>
        <div className="hero-cta">
          <Link className="btn primary big" to={session ? '/label' : '/login'}>Start labelling</Link>
          <Link className="btn ghost big" to="/guide">Hear the sound guide</Link>
        </div>
        <p className="muted small-print">18+ only: some clips contain intimate vocal sounds (breathing, kissing, moaning).</p>
      </section>

      <section className="stat-row">
        <div className="stat"><b>{fmt(stats?.labels)}</b><span>labels given</span></div>
        <div className="stat"><b>{fmt(stats?.labellers)}</b><span>listeners</span></div>
        <div className="stat"><b>{fmt(stats?.clips_complete)}</b><span>clips fully labelled</span></div>
        <div className="stat"><b>{fmt(stats?.clips)}</b><span>clips in the queue</span></div>
      </section>

      <section className="two-col">
        <div className="card">
          <h2>How it works</h2>
          <ol className="steps">
            <li><b>Listen.</b> One short clip at a time, headphones recommended.</li>
            <li><b>Tick what you hear.</b> Usually more than one sound; “something else” lets you describe it.</li>
            <li><b>Blind by design.</b> You never see anyone else’s answer or what a model thinks, so every label is your own.</li>
            <li><b>It adds up.</b> Answers are combined across listeners; your profile tracks every clip you label.</li>
          </ol>
        </div>
        <div className="card">
          <div className="card-head">
            <h2>Top listeners · 30 days</h2>
            <Link to="/leaderboard" className="muted">All →</Link>
          </div>
          {top.length === 0 ? (
            <p className="muted">No labels yet — be the first.</p>
          ) : (
            <ol className="mini-board">
              {top.map((r) => (
                <li key={r.username}>
                  <span className="rank">{r.rank}</span>
                  <Avatar name={r.username} url={r.avatar_url} size={24} />
                  <Link to={`/u/${r.username}`}>{r.display_name || r.username}</Link>
                  <span className="count">{r.labelled.toLocaleString()}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </section>
    </main>
  )
}
