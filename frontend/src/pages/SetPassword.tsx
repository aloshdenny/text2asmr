import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'

export default function SetPassword() {
  const { session, profile, loading } = useAuth()
  const navigate = useNavigate()
  const [password, setPassword] = useState('')
  const [again, setAgain] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (loading) return <main className="narrow"><p className="muted">Loading…</p></main>
  if (!session) return <Navigate to="/login" replace />

  const save = async (e: FormEvent) => {
    e.preventDefault()
    if (password.length < 8) return setError('Use at least 8 characters.')
    if (password !== again) return setError('The two passwords differ.')
    setBusy(true)
    const { error } = await supabase.auth.updateUser({ password })
    setBusy(false)
    if (error) return setError(error.message)
    navigate(profile ? `/u/${profile.username}` : '/welcome', { replace: true })
  }
  return (
    <main className="narrow auth">
      <div className="card">
        <h1>Choose a password</h1>
        <p className="muted">For {session.user.email}. You’ll use it to sign in from now on.</p>
        <form onSubmit={save} className="stack">
          <label>
            New password
            <input type="password" autoComplete="new-password" minLength={8} required value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
          <label>
            Same again
            <input type="password" autoComplete="new-password" minLength={8} required value={again} onChange={(e) => setAgain(e.target.value)} />
          </label>
          <button className="btn primary" disabled={busy}>{busy ? 'Saving…' : 'Save password'}</button>
          {error && <p className="error">{error}</p>}
        </form>
      </div>
    </main>
  )
}
