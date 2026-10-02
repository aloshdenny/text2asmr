import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import type { EmailOtpType } from '@supabase/supabase-js'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, ButtonLink, Field, Input, Loading, PasswordInput } from '../components/ui'
import { useAuth } from '../lib/auth'
import { friendlyError, supabase } from '../lib/supabase'

/** Ask for a reset link (needs working email delivery on the project). */
export function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    const { error } = await supabase.auth.resetPasswordForEmail(email.trim())
    setBusy(false)
    if (error) setError(friendlyError(error.message))
    else setSent(true)
  }
  return (
    <AuthLayout>
      <form onSubmit={submit} className="space-y-5 rounded-xl border border-hairline bg-surface p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-ink">Reset your password</h1>
          <p className="text-sm text-ink-muted">{sent ? `If ${email.trim()} has an account, a reset link is on its way.` : 'We’ll email you a link to choose a new one.'}</p>
        </div>
        {!sent && (
          <>
            <Field label="Email" htmlFor="email">
              <Input id="email" type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} />
            </Field>
            {error && <Alert>{error}</Alert>}
            <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Sending…' : 'Send reset link'}</Button>
          </>
        )}
        <p className="text-center text-sm text-ink-muted"><Link to="/login" className="font-medium text-brand hover:underline">Back to log in</Link></p>
      </form>
    </AuthLayout>
  )
}

/** Email links land here: the one-time token hash is verified through the site's proxy. */
export function AuthConfirm() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const [error, setError] = useState('')
  const once = useRef(false)
  useEffect(() => {
    if (once.current) return
    once.current = true
    const type = (params.get('type') ?? 'email') as EmailOtpType
    supabase.auth.verifyOtp({ token_hash: params.get('token_hash') ?? '', type }).then(({ error }) => {
      if (error) setError(error.message)
      else navigate(type === 'recovery' ? '/reset-password' : '/welcome', { replace: true })
    })
  }, [params, navigate])
  return (
    <AuthLayout>
      <div className="space-y-4 rounded-xl border border-hairline bg-surface p-6">
        {error ? (
          <>
            <h1 className="text-xl font-semibold text-ink">That link didn’t work</h1>
            <p className="text-sm text-ink-muted">{error}. Links work once and expire after an hour.</p>
            <ButtonLink to="/login" className="w-full">Back to log in</ButtonLink>
          </>
        ) : (
          <Loading />
        )}
      </div>
    </AuthLayout>
  )
}

export function ResetPassword() {
  const { session, loading } = useAuth()
  const navigate = useNavigate()
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (loading) return <AuthLayout><Loading /></AuthLayout>
  if (!session) return <Navigate to="/login" replace />
  const save = async (e: FormEvent) => {
    e.preventDefault()
    if (password.length < 8) return setError('Use at least 8 characters.')
    setBusy(true)
    const { error } = await supabase.auth.updateUser({ password })
    setBusy(false)
    if (error) return setError(error.message)
    navigate('/dashboard', { replace: true })
  }
  return (
    <AuthLayout>
      <form onSubmit={save} className="space-y-5 rounded-xl border border-hairline bg-surface p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-ink">Choose a new password</h1>
          <p className="text-sm text-ink-muted">For {session.user.email}.</p>
        </div>
        <Field label="New password" htmlFor="password" hint="At least 8 characters.">
          <PasswordInput id="password" required minLength={8} autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </Field>
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Saving…' : 'Save password'}</Button>
      </form>
    </AuthLayout>
  )
}
