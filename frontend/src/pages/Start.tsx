import { useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Loading } from '../components/ui'
import { useAuth } from '../lib/auth'
import { friendlyError, supabase } from '../lib/supabase'

/** Label without an account: confirm 18+, get a guest session, go straight to the clips. A guest is a real (anonymous)
 *  user, so check clips, retests and per-person trust work as for everyone; saving progress later keeps the labels. */
export default function Start() {
  const { session, profile, loading, refreshProfile } = useAuth()
  const navigate = useNavigate()
  const [adult, setAdult] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  if (loading) return <AuthLayout><Loading /></AuthLayout>
  if (session && profile?.onboarded) return <Navigate to="/label" replace />
  if (session && !session.user.is_anonymous) return <Navigate to="/welcome" replace />

  const start = async () => {
    setError('')
    if (!adult) return setError('You need to be 18 or older to label these clips.')
    setBusy(true)
    if (!session) {
      const { error } = await supabase.auth.signInAnonymously()
      if (error) {
        setBusy(false)
        return setError(/anonymous/i.test(error.message) ? 'Guest labelling isn’t switched on yet. Create an account to start.' : friendlyError(error.message))
      }
    }
    const { error } = await supabase.rpc('start_guest')
    if (error) {
      setBusy(false)
      return setError(friendlyError(error.message))
    }
    await refreshProfile()
    setBusy(false)
    navigate('/label', { replace: true })
  }

  return (
    <AuthLayout>
      <div className="space-y-5 rounded-xl border border-hairline bg-surface p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-ink">Start labelling</h1>
          <p className="text-sm text-ink-muted">No account needed. Listen to a short clip, tick what you hear, next clip. You can save your progress and join the leaderboard any time.</p>
        </div>
        <label className="flex items-start gap-2.5 text-sm text-ink-secondary">
          <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[var(--brand)]" />
          <span>I’m 18 or older. Some clips contain intimate vocal sounds, and my labels are shared with the text2asmr project.</span>
        </label>
        {error && <Alert>{error}</Alert>}
        <Button variant="primary" disabled={busy} className="w-full" onClick={start}>{busy ? 'Starting…' : 'Start labelling'}</Button>
        <p className="text-center text-sm text-ink-muted">
          Want to be on the leaderboard? <Link to="/signup" className="font-medium text-brand hover:underline">Create an account</Link>
        </p>
      </div>
    </AuthLayout>
  )
}
