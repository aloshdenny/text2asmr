import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import type { EmailOtpType } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'

/** Landing page for email links (confirm sign-up, set / reset password): the links carry a one-time token hash that
 *  is verified here through the proxy, so emails never point at the Supabase project itself. */
export default function AuthConfirm() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const [error, setError] = useState('')
  const once = useRef(false)
  useEffect(() => {
    if (once.current) return
    once.current = true
    const token_hash = params.get('token_hash') ?? ''
    const type = (params.get('type') ?? 'email') as EmailOtpType
    supabase.auth.verifyOtp({ token_hash, type }).then(({ error }) => {
      if (error) setError(error.message)
      else navigate(type === 'recovery' ? '/set-password' : '/welcome', { replace: true })
    })
  }, [params, navigate])
  return (
    <main className="narrow auth">
      <div className="card">
        {error ? (
          <>
            <h1>That link didn’t work</h1>
            <p className="muted">{error}. Links work once and expire after an hour.</p>
            <Link className="btn" to="/login">Back to sign in</Link>
          </>
        ) : (
          <p className="muted">Checking your link…</p>
        )}
      </div>
    </main>
  )
}
