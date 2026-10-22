import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Field, Input, Loading } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useUserStats } from '../lib/stats'
import { friendlyError, supabase } from '../lib/supabase'

/** A guest keeps their labels by adding an email to the same (anonymous) account. Once they confirm it, /welcome asks
 *  for a username and password, and the first profile save makes them a full labeller on the leaderboard. */
export default function SaveProgress() {
  const { session, profile, loading } = useAuth()
  const { stats } = useUserStats(profile?.username)
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [sent, setSent] = useState(false)
  const [busy, setBusy] = useState(false)
  if (loading) return <AuthLayout><Loading /></AuthLayout>
  if (!session) return <Navigate to="/signup" replace />
  if (!session.user.is_anonymous) return <Navigate to="/welcome" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    const { error } = await supabase.auth.updateUser({ email: email.trim() }, { emailRedirectTo: `${window.location.origin}/welcome` })
    setBusy(false)
    if (error) {
      return setError(/already|registered|exists/i.test(error.message)
        ? 'That email already has an account. Log in to it instead (your guest labels still count for the project).'
        : friendlyError(error.message))
    }
    setSent(true)
  }

  if (sent)
    return (
      <AuthLayout>
        <div className="space-y-3">
          <h1 className="text-center text-[26px] font-black tracking-tight text-ink">Confirm your email</h1>
          <p className="text-sm text-ink-secondary">We sent a link to <span className="font-medium text-ink">{email.trim()}</span>. Open it on this device to pick a username and password. Your labels stay with you.</p>
        </div>
      </AuthLayout>
    )

  const n = stats?.labelled ?? 0
  return (
    <AuthLayout>
      <form onSubmit={submit} className="space-y-5">
        <div className="space-y-1.5 text-center">
          <h1 className="text-center text-[26px] font-black tracking-tight text-ink">Save your progress</h1>
          <p className="text-sm text-ink-muted">
            {n > 0 ? `Keep your ${n.toLocaleString()} labelled clips` : 'Keep your labels'}, come back from any device and join the leaderboard. Add your email and we’ll send a link to finish.
          </p>
        </div>
        <Field label="Email" htmlFor="email">
          <Input id="email" type="email" required autoComplete="email" autoCapitalize="none" spellCheck={false} value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Sending…' : 'Send link'}</Button>
      </form>
    </AuthLayout>
  )
}
