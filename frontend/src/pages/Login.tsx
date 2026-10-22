import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Field, Input, PasswordInput, ButtonLink } from '../components/ui'
import { useAuth } from '../lib/auth'
import { friendlyError, supabase } from '../lib/supabase'

export default function Login() {
  const { session, loading } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [unclaimed, setUnclaimed] = useState(false)
  const [busy, setBusy] = useState(false)
  if (!loading && session && !session.user.is_anonymous) return <Navigate to="/dashboard" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setUnclaimed(false)
    setBusy(true)
    const addr = email.trim()
    const { error } = await supabase.auth.signInWithPassword({ email: addr, password })
    if (error) {
      // a kit labeller whose profile is still waiting has no account yet: point them at sign-up
      const { data: status } = await supabase.rpc('account_status', { p_email: addr })
      setBusy(false)
      if (status === 'unclaimed') return setUnclaimed(true)
      return setError(error.message === 'Invalid login credentials' ? 'Wrong email or password.' : friendlyError(error.message))
    }
    setBusy(false)
    navigate('/dashboard')
  }

  return (
    <AuthLayout action={<ButtonLink to="/signup" variant="secondary" size="sm">Sign up</ButtonLink>}>
      <form onSubmit={submit} className="space-y-5">
        <div className="space-y-1.5 text-center">
          <h1 className="text-center text-[26px] font-black tracking-tight text-ink">Log in</h1>
          <p className="text-sm text-ink-muted">Welcome back.</p>
        </div>
        <Field label="Email" htmlFor="email" quiet>
          <Input id="email" type="email" placeholder="Email" required autoComplete="email" autoCapitalize="none" spellCheck={false} value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        <Field label="Password" htmlFor="password" quiet>
          <PasswordInput id="password" placeholder="Password" required autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </Field>
        <div className="-mt-2 text-right">
          <Link to="/forgot-password" className="text-xs text-ink-muted transition-colors hover:text-ink">Forgot your password?</Link>
        </div>
        {unclaimed && (
          <Alert tone="brand">
            Your Ear Check labels are waiting for this email, but there’s no account yet.{' '}
            <Link to={`/signup?email=${encodeURIComponent(email.trim())}`} className="font-medium text-brand hover:underline">Sign up with it</Link> to claim them.
          </Alert>
        )}
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Logging in…' : 'Log in'}</Button>
        <p className="text-center text-sm text-ink-muted">
          New here? <Link to="/signup" className="font-medium text-brand hover:underline">Create an account</Link>
        </p>
      </form>
    </AuthLayout>
  )
}
