import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'

const USERNAME = /^[a-z0-9_]{3,20}$/

export default function Welcome() {
  const { session, profile, loading, refreshProfile } = useAuth()
  const navigate = useNavigate()
  const meta = session?.user.user_metadata ?? {}
  const [username, setUsername] = useState(String(meta.user_name ?? meta.preferred_username ?? '').toLowerCase().replace(/[^a-z0-9_]/g, '').slice(0, 20))
  const [displayName, setDisplayName] = useState(String(meta.full_name ?? meta.name ?? '').slice(0, 40))
  const [adult, setAdult] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (loading) return <main className="narrow"><p className="muted">Loading…</p></main>
  if (!session) return <Navigate to="/login" replace />
  if (profile) return <Navigate to="/label" replace />

  const save = async (e: FormEvent) => {
    e.preventDefault()
    if (!USERNAME.test(username)) return setError('Usernames are 3–20 characters: lowercase letters, digits and _')
    if (!adult) return setError('You need to be 18 or older to label these clips.')
    setBusy(true)
    const avatar = typeof meta.avatar_url === 'string' && meta.avatar_url.startsWith('https://') ? meta.avatar_url : null
    const { error } = await supabase
      .from('profiles')
      .insert({ id: session.user.id, username, display_name: displayName.trim() || null, avatar_url: avatar })
    setBusy(false)
    if (error) return setError(error.code === '23505' ? 'That username is taken — try another.' : error.message)
    await refreshProfile()
    navigate('/label')
  }
  return (
    <main className="narrow auth">
      <div className="card">
        <h1>Welcome, listener</h1>
        <p className="muted">Pick the name your contributions appear under on the leaderboard.</p>
        <form onSubmit={save} className="stack">
          <label>
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value.toLowerCase())} placeholder="quiet_ears" required maxLength={20} />
          </label>
          <label>
            Display name <span className="muted">(optional)</span>
            <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} maxLength={40} />
          </label>
          <label className="check">
            <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} />
            <span>I am 18 or older. I understand some clips contain intimate vocal sounds, and that my labels are shared with the text2asmr project under my username.</span>
          </label>
          <button className="btn primary" disabled={busy}>{busy ? 'Saving…' : 'Start labelling'}</button>
          {error && <p className="error">{error}</p>}
        </form>
      </div>
    </main>
  )
}
