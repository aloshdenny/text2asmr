import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Field, Input, PasswordInput } from '../components/ui'
import { useAuth } from '../lib/auth'
import { friendlyError, supabase } from '../lib/supabase'

export default function Signup() {
  const { session, loading, refreshProfile } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [email, setEmail] = useState(params.get('email') ?? '')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [sent, setSent] = useState(false)
  const [busy, setBusy] = useState(false)
  if (!loading && session) return <Navigate to="/welcome" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    if (password.length < 8) return setError('Use at least 8 characters for your password.')
    setBusy(true)
    const { data, error } = await supabase.auth.signUp({ email: email.trim(), password })
    setBusy(false)
    if (error) {
      return setError(/already registered|already exists/i.test(error.message) ? 'There’s already an account for this email. Log in instead.' : friendlyError(error.message))
    }
    if (!data.session) return setSent(true) // the project asks new accounts to confirm their email first
    // signing up with a kit labeller's email claims their profile (and labels) on the way in
    await refreshProfile()
    navigate('/welcome')
  }

  if (sent)
    return (
      <AuthLayout>
        <div className="space-y-3 rounded-xl border border-hairline bg-surface p-6">
          <h1 className="text-xl font-semibold text-ink">Confirm your email</h1>
          <p className="text-sm text-ink-secondary">We sent a link to <span className="font-medium text-ink">{email.trim()}</span>. Open it to finish creating your account.</p>
        </div>
      </AuthLayout>
    )

  return (
    <AuthLayout>
      <form onSubmit={submit} className="space-y-5 rounded-xl border border-hairline bg-surface p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-ink">Create your account</h1>
          <p className="text-sm text-ink-muted">Labelled with us before? Use the same email and your labels come with you.</p>
        </div>
        <Field label="Email" htmlFor="email">
          <Input id="email" type="email" required autoComplete="email" autoCapitalize="none" spellCheck={false} value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        <Field label="Password" htmlFor="password" hint="At least 8 characters.">
          <PasswordInput id="password" required minLength={8} autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </Field>
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Creating account…' : 'Create account'}</Button>
        <p className="text-center text-sm text-ink-muted">
          Already have an account? <Link to="/login" className="font-medium text-brand hover:underline">Log in</Link>
        </p>
      </form>
    </AuthLayout>
  )
}
