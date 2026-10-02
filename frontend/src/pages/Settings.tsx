import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import AvatarPicker from '../components/AvatarPicker'
import { AppLayout } from '../components/Layouts'
import { Alert, Button, Card, CardHeader, Field, Input, PageHeader } from '../components/ui'
import { useAuth, type Profile } from '../lib/auth'
import { supabase } from '../lib/supabase'

const USERNAME = /^[a-z0-9_]{3,20}$/

export default function Settings() {
  const { session, profile } = useAuth()
  if (!session || !profile) return null
  return <SettingsForm key={profile.id} profile={profile} uid={session.user.id} />
}

function SettingsForm({ profile, uid }: { profile: Profile; uid: string }) {
  const { refreshProfile } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState(profile.username)
  const [displayName, setDisplayName] = useState(profile.display_name ?? '')
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [confirm, setConfirm] = useState('')

  const setAvatar = async (avatar_url: string | null) => {
    const { error } = await supabase.from('profiles').update({ avatar_url }).eq('id', uid)
    if (error) return setMsg({ ok: false, text: error.message })
    await refreshProfile()
  }

  const saveNames = async (e: FormEvent) => {
    e.preventDefault()
    const u = username.trim().toLowerCase()
    if (!USERNAME.test(u)) return setMsg({ ok: false, text: 'Usernames are 3–20 characters: lowercase letters, digits and _.' })
    setBusy(true)
    const { error } = await supabase.from('profiles').update({ username: u, display_name: displayName.trim() || null }).eq('id', uid)
    setBusy(false)
    if (error) return setMsg({ ok: false, text: error.code === '23505' ? 'That username is taken — try another.' : error.message })
    await refreshProfile()
    setMsg({ ok: true, text: 'Saved.' })
  }

  const deleteAccount = async () => {
    setBusy(true)
    const { data } = await supabase.storage.from('avatars').list(uid)
    if (data?.length) await supabase.storage.from('avatars').remove(data.map((f) => `${uid}/${f.name}`))
    const { error } = await supabase.rpc('delete_account')
    if (error) {
      setBusy(false)
      return setMsg({ ok: false, text: error.message })
    }
    navigate('/', { replace: true })
    await supabase.auth.signOut({ scope: 'local' })
  }

  return (
    <AppLayout width="narrow">
      <PageHeader title="Edit profile" description="How you appear on the leaderboard and your public page." />
      <div className="space-y-3">
        <Card>
          <CardHeader title="Picture" />
          <AvatarPicker uid={uid} name={profile.username} value={profile.avatar_url} onChange={setAvatar} />
        </Card>

        <Card>
          <CardHeader title="Name" />
          <form onSubmit={saveNames} className="space-y-4">
            <Field label="Username" htmlFor="username" hint="3–20 characters: lowercase letters, digits and _.">
              <Input id="username" required maxLength={20} autoCapitalize="none" spellCheck={false} value={username} onChange={(e) => setUsername(e.target.value.toLowerCase())} />
            </Field>
            <Field label={<>Display name <span className="font-normal text-ink-muted">(optional)</span></>} htmlFor="display">
              <Input id="display" maxLength={40} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
            </Field>
            {msg && (msg.ok ? <p className="text-sm text-ok">{msg.text}</p> : <Alert>{msg.text}</Alert>)}
            <div className="flex justify-end">
              <Button type="submit" variant="primary" disabled={busy}>Save changes</Button>
            </div>
          </form>
        </Card>

        <Card>
          <CardHeader
            title="Delete account"
            description="Removes your sign-in and takes you off the leaderboard and profile pages. The clips you labelled stay in the dataset, without your name. This can’t be undone."
          />
          <div className="space-y-3">
            <Field label={<>Type <span className="font-mono">{profile.username}</span> to confirm</>} htmlFor="confirm">
              <Input id="confirm" autoComplete="off" value={confirm} onChange={(e) => setConfirm(e.target.value)} />
            </Field>
            <div className="flex justify-end">
              <Button variant="danger" disabled={busy || confirm !== profile.username} onClick={deleteAccount}>Delete my account</Button>
            </div>
          </div>
        </Card>
      </div>
    </AppLayout>
  )
}
