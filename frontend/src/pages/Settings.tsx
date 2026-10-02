import { useState, type ChangeEvent, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import Avatar from '../components/Avatar'
import { useAuth, type Profile } from '../lib/auth'
import { supabase } from '../lib/supabase'

const USERNAME = /^[a-z0-9_]{3,20}$/

/** Square-crop and shrink a picture to 256 px WebP in the browser (also drops photo metadata). */
async function toAvatar(file: File): Promise<Blob> {
  const img = await createImageBitmap(file)
  const side = Math.min(img.width, img.height)
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 256
  canvas.getContext('2d')!.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, 256, 256)
  return new Promise((ok, fail) => canvas.toBlob((b) => (b ? ok(b) : fail(new Error('Could not read that image'))), 'image/webp', 0.85))
}

export default function Settings() {
  const { session, profile, loading } = useAuth()
  if (loading) return <main className="narrow"><p className="muted">Loading…</p></main>
  if (!session) return <Navigate to="/login" replace />
  if (!profile) return <Navigate to="/welcome" replace />
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

  const saveNames = async (e: FormEvent) => {
    e.preventDefault()
    const u = username.trim().toLowerCase()
    if (!USERNAME.test(u)) return setMsg({ ok: false, text: 'Usernames are 3–20 characters: lowercase letters, digits and _' })
    setBusy(true)
    const { error } = await supabase.from('profiles').update({ username: u, display_name: displayName.trim() || null }).eq('id', uid)
    setBusy(false)
    if (error) return setMsg({ ok: false, text: error.code === '23505' ? 'That username is taken — try another.' : error.message })
    await refreshProfile()
    setMsg({ ok: true, text: 'Saved.' })
  }

  const ownAvatars = async () => {
    const { data } = await supabase.storage.from('avatars').list(uid)
    return (data ?? []).map((f) => `${uid}/${f.name}`)
  }

  const upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusy(true)
    setMsg(null)
    try {
      const blob = await toAvatar(file)
      const old = await ownAvatars()
      const path = `${uid}/${Date.now()}.webp`
      const { error } = await supabase.storage.from('avatars').upload(path, blob, { contentType: 'image/webp' })
      if (error) throw error
      const { error: e2 } = await supabase.from('profiles').update({ avatar_url: `avatars/${path}` }).eq('id', uid)
      if (e2) throw e2
      if (old.length) await supabase.storage.from('avatars').remove(old)
      await refreshProfile()
      setMsg({ ok: true, text: 'Picture updated.' })
    } catch (err) {
      setMsg({ ok: false, text: err instanceof Error ? err.message : String(err) })
    }
    setBusy(false)
  }

  const removeAvatar = async () => {
    setBusy(true)
    const old = await ownAvatars()
    await supabase.from('profiles').update({ avatar_url: null }).eq('id', uid)
    if (old.length) await supabase.storage.from('avatars').remove(old)
    await refreshProfile()
    setBusy(false)
  }

  const deleteAccount = async () => {
    setBusy(true)
    const old = await ownAvatars()
    if (old.length) await supabase.storage.from('avatars').remove(old)
    const { error } = await supabase.rpc('delete_account')
    if (error) {
      setBusy(false)
      return setMsg({ ok: false, text: error.message })
    }
    navigate('/', { replace: true })
    await supabase.auth.signOut({ scope: 'local' })
  }

  return (
    <main className="narrow">
      <h1>Your profile</h1>
      <section className="card settings">
        <h2>Picture</h2>
        <div className="avatar-row">
          <Avatar name={profile.username} url={profile.avatar_url} size={72} />
          <label className="btn">
            Upload a picture
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={upload} disabled={busy} hidden />
          </label>
          {profile.avatar_url && <button className="btn ghost" onClick={removeAvatar} disabled={busy}>Remove</button>}
        </div>
      </section>

      <section className="card settings">
        <h2>Name</h2>
        <form onSubmit={saveNames} className="stack">
          <label>
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value.toLowerCase())} maxLength={20} required />
          </label>
          <label>
            Display name <span className="muted">(optional)</span>
            <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} maxLength={40} />
          </label>
          <button className="btn primary" disabled={busy}>Save</button>
        </form>
      </section>
      {msg && <p className={msg.ok ? 'ok-msg' : 'error'}>{msg.text}</p>}

      <section className="card settings danger">
        <h2>Delete account</h2>
        <p className="muted">
          Your sign-in and your place on the leaderboard and profile pages are removed. The clips you labelled stay in
          the dataset, without your name. This can’t be undone.
        </p>
        <label className="stack">
          <span>Type <b>{profile.username}</b> to confirm</span>
          <input value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="off" />
        </label>
        <button className="btn danger" disabled={busy || confirm !== profile.username} onClick={deleteAccount}>
          Delete my account
        </button>
      </section>
    </main>
  )
}
