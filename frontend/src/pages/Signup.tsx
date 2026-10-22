import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Field, Input, PasswordInput, ButtonLink } from '../components/ui'
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
  if (!loading && session?.user.is_anonymous) return <Navigate to="/save-progress" replace />
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
        <div className="space-y-3">
          <h1 className="text-center text-[26px] font-black tracking-tight text-ink">Confirm your email</h1>
          <p className="text-sm text-ink-secondary">We sent a link to <span className="font-medium text-ink">{email.trim()}</span>. Open it to finish creating your account.</p>
        </div>
      </AuthLayout>
    )

  return (
    <AuthLayout action={<ButtonLink to="/login" variant="secondary" size="sm">Log in</ButtonLink>}>
      <form onSubmit={submit} className="space-y-5">
        <div className="space-y-1.5 text-center">
          <h1 className="text-center text-[26px] font-black tracking-tight text-ink">Create your account</h1>
          <p className="text-sm text-ink-muted">Labelled with us before? Use the same email and your labels come with you.</p>
        </div>
        <Field label="Email" htmlFor="email" quiet>
          <Input id="email" type="email" placeholder="Email" required autoComplete="email" autoCapitalize="none" spellCheck={false} value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        <Field label="Password" htmlFor="password" quiet>
          <PasswordInput id="password" placeholder="Password (at least 8 characters)" required minLength={8} autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </Field>
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Creating account…' : 'Create account'}</Button>
        <p className="text-xs text-ink-muted">To spot duplicate accounts, the site keeps scrambled codes for this browser and network. They can’t be turned back into your IP address and are deleted after 180 days without use.</p>
        <p className="text-center text-sm text-ink-muted">
          Already have an account? <Link to="/login" className="font-medium text-brand hover:underline">Log in</Link>
        </p>
      </form>
    </AuthLayout>
  )
}
