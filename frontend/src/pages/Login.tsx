import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'

type Mode = 'signin' | 'signup'
type Notice = { kind: 'claim' | 'sent' | 'confirm'; email: string } | null

export default function Login() {
  const { session, profile, loading } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [notice, setNotice] = useState<Notice>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (!loading && session) return <Navigate to={profile ? '/label' : '/welcome'} replace />

  const emailSetPassword = async (to: string) => {
    setBusy(true)
    const { error } = await supabase.auth.resetPasswordForEmail(to)
    setBusy(false)
    if (error) return setError(error.message)
    setNotice({ kind: 'sent', email: to })
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    const addr = email.trim()
    setBusy(true)
    // people who labelled the Ear Check kits before the site already have an account, without a password
    const { data: status } = await supabase.rpc('account_status', { p_email: addr })
    if (status === 'needs_password') {
      setBusy(false)
      return setNotice({ kind: 'claim', email: addr })
    }
    if (mode === 'signin') {
      const { error } = await supabase.auth.signInWithPassword({ email: addr, password })
      setBusy(false)
      if (error) return setError(error.message === 'Invalid login credentials' ? 'Wrong email or password.' : error.message)
      navigate('/label')
    } else {
      const { data, error } = await supabase.auth.signUp({ email: addr, password })
      setBusy(false)
      if (error) return setError(error.message)
      if (data.session) navigate('/welcome')
      else setNotice({ kind: 'confirm', email: addr })
    }
  }

  if (notice)
    return (
      <main className="narrow auth">
        <div className="card">
          {notice.kind === 'claim' && (
            <>
              <h1>Welcome back</h1>
              <p>
                <b>{notice.email}</b> already has an account here, with the clips you labelled in the Ear Check kits
                credited to it — but no password yet. Set one to get into your profile.
              </p>
              <div className="stack">
                <button className="btn primary" disabled={busy} onClick={() => emailSetPassword(notice.email)}>
                  {busy ? 'Sending…' : 'Email me a link to set my password'}
                </button>
                {error && <p className="error">{error}</p>}
              </div>
            </>
          )}
          {notice.kind === 'sent' && (
            <>
              <h1>Check your inbox</h1>
              <p>We sent a link to <b>{notice.email}</b>. Open it to choose your password.</p>
            </>
          )}
          {notice.kind === 'confirm' && (
            <>
              <h1>Confirm your email</h1>
              <p>We sent a link to <b>{notice.email}</b>. Open it to finish creating your account.</p>
            </>
          )}
          <button className="btn ghost small back" onClick={() => setNotice(null)}>← Back</button>
        </div>
      </main>
    )

  return (
    <main className="narrow auth">
      <div className="card">
        <div className="tabs wide" role="tablist">
          <button role="tab" aria-selected={mode === 'signin'} className={mode === 'signin' ? 'on' : ''} onClick={() => setMode('signin')}>Sign in</button>
          <button role="tab" aria-selected={mode === 'signup'} className={mode === 'signup' ? 'on' : ''} onClick={() => setMode('signup')}>Create account</button>
        </div>
        <form onSubmit={submit} className="stack">
          <label>
            Email
            <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </label>
          <label>
            Password
            <input
              type="password"
              required
              minLength={8}
              autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === 'signup' ? 'at least 8 characters' : ''}
            />
          </label>
          <button className="btn primary" disabled={busy}>
            {busy ? 'One moment…' : mode === 'signin' ? 'Sign in' : 'Create account'}
          </button>
          {error && <p className="error">{error}</p>}
        </form>
        {mode === 'signin' && (
          <button
            className="link-btn"
            onClick={() => (email.trim() ? emailSetPassword(email.trim()) : setError('Enter your email first.'))}
          >
            Forgot your password?
          </button>
        )}
      </div>
    </main>
  )
}
