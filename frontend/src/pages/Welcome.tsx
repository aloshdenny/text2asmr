import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import AvatarPicker from '../components/AvatarPicker'
import { AuthLayout } from '../components/Layouts'
import { Alert, Button, Field, Input, Loading } from '../components/ui'
import { useAuth, type Profile } from '../lib/auth'
import { useUserStats } from '../lib/stats'
import { friendlyError, supabase } from '../lib/supabase'

const USERNAME = /^[a-z0-9_]{3,20}$/

/** Onboarding, once per account: username, optional picture, 18+ confirmation. A kit labeller who just signed up
 *  already has a profile (claimed by email) -- they see their labels and can change the username they were given. */
export default function Welcome() {
  const { session, profile, loading } = useAuth()
  if (loading) return <AuthLayout><Loading /></AuthLayout>
  if (!session) return <Navigate to="/signup" replace />
  if (profile?.onboarded) return <Navigate to="/dashboard" replace />
  return <WelcomeForm key={profile?.id ?? 'new'} uid={session.user.id} profile={profile} />
}

function WelcomeForm({ uid, profile }: { uid: string; profile: Profile | null }) {
  const { refreshProfile } = useAuth()
  const navigate = useNavigate()
  const { stats } = useUserStats(profile?.username)
  const [username, setUsername] = useState(profile?.username ?? '')
  const [displayName, setDisplayName] = useState(profile?.display_name ?? '')
  const [avatar, setAvatar] = useState<string | null>(profile?.avatar_url ?? null)
  const [adult, setAdult] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const save = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    const u = username.trim().toLowerCase()
    if (!USERNAME.test(u)) return setError('Usernames are 3 to 20 characters: lowercase letters, digits and _.')
    if (!adult) return setError('You need to be 18 or older to label these clips.')
    setBusy(true)
    const fields = { username: u, display_name: displayName.trim() || null, avatar_url: avatar }
    const { error } = profile
      ? await supabase.from('profiles').update({ ...fields, onboarded: true }).eq('id', uid)
      : await supabase.from('profiles').insert({ id: uid, ...fields })
    setBusy(false)
    if (error) return setError(error.code === '23505' ? 'That username is taken. Try another.' : friendlyError(error.message))
    await refreshProfile()
    navigate('/dashboard', { replace: true })
  }

  return (
    <AuthLayout>
      <form onSubmit={save} className="space-y-5 rounded-xl border border-hairline bg-surface p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold text-ink">{profile ? 'Welcome back' : 'Set up your profile'}</h1>
          <p className="text-sm text-ink-muted">
            {profile
              ? stats
                ? `Your ${stats.labelled.toLocaleString()} labelled clips from the Ear Check kits are on this account. Keep your username or pick a new one.`
                : 'Your labels from the Ear Check kits are on this account.'
              : 'This is how you appear on the leaderboard.'}
          </p>
        </div>
        <Field label="Username" htmlFor="username" hint="3 to 20 characters: lowercase letters, digits and _.">
          <Input id="username" required maxLength={20} autoCapitalize="none" spellCheck={false} value={username} placeholder="quiet_ears" onChange={(e) => setUsername(e.target.value.toLowerCase())} />
        </Field>
        <Field label={<>Display name <span className="font-normal text-ink-muted">(optional)</span></>} htmlFor="display">
          <Input id="display" maxLength={40} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
        </Field>
        <Field label={<>Profile picture <span className="font-normal text-ink-muted">(optional)</span></>}>
          <AvatarPicker uid={uid} name={username || displayName} value={avatar} onChange={setAvatar} />
        </Field>
        <label className="flex items-start gap-2.5 text-sm text-ink-secondary">
          <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[var(--brand)]" />
          <span>I’m 18 or older. Some clips contain intimate vocal sounds, and my labels are shared with the text2asmr project under my username.</span>
        </label>
        {error && <Alert>{error}</Alert>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">{busy ? 'Saving…' : 'Continue'}</Button>
      </form>
    </AuthLayout>
  )
}
