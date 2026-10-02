import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { GITHUB_AUTH, supabase } from '../lib/supabase'

export default function Login() {
  const { session, profile, loading } = useAuth()
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (!loading && session) return <Navigate to={profile ? '/label' : '/welcome'} replace />

  const redirectTo = `${window.location.origin}/label`
  const send = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    const { error } = await supabase.auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo } })
    setBusy(false)
    if (error) setError(error.message)
    else setSent(true)
  }
  return (
    <main className="narrow auth">
      <div className="card">
        <h1>Sign in to label</h1>
        {sent ? (
          <p>
            Check <b>{email}</b> for a sign-in link. It opens Ear Check signed in — no password needed.
          </p>
        ) : (
          <>
            <p className="muted">We email you a one-time sign-in link. Your email is never shown to anyone.</p>
            <form onSubmit={send} className="stack">
              <label>
                Email
                <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
              </label>
              <button className="btn primary" disabled={busy}>{busy ? 'Sending…' : 'Email me a sign-in link'}</button>
            </form>
            {GITHUB_AUTH && (
              <>
                <div className="or"><span>or</span></div>
                <button className="btn" onClick={() => supabase.auth.signInWithOAuth({ provider: 'github', options: { redirectTo } })}>
                  Continue with GitHub
                </button>
              </>
            )}
            {error && <p className="error">{error}</p>}
          </>
        )}
      </div>
    </main>
  )
}
