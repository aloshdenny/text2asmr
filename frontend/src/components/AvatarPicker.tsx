import { useState, type ChangeEvent } from 'react'
import { supabase } from '../lib/supabase'
import Avatar from './Avatar'
import { Button, Spinner } from './ui'

/** Square-crop and shrink a picture to 256 px WebP in the browser (also drops photo metadata). */
async function toAvatar(file: File): Promise<Blob> {
  const img = await createImageBitmap(file)
  const side = Math.min(img.width, img.height)
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 256
  canvas.getContext('2d')!.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, 256, 256)
  return new Promise((ok, fail) => canvas.toBlob((b) => (b ? ok(b) : fail(new Error('Could not read that image'))), 'image/webp', 0.85))
}

/** Upload / remove a profile picture. Stores it under avatars/<uid>/ and hands the new value (or null) to onChange;
 *  the caller saves it on the profile. Older pictures in the folder are removed once a new one is saved. */
export default function AvatarPicker({ uid, name, value, onChange }: { uid: string; name: string; value: string | null; onChange: (v: string | null) => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const others = async (keep?: string) => {
    const { data } = await supabase.storage.from('avatars').list(uid)
    return (data ?? []).map((f) => `${uid}/${f.name}`).filter((p) => `avatars/${p}` !== keep)
  }

  const upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const path = `${uid}/${Date.now()}.webp`
      const { error } = await supabase.storage.from('avatars').upload(path, await toAvatar(file), { contentType: 'image/webp' })
      if (error) throw error
      const next = `avatars/${path}`
      onChange(next)
      const old = await others(next)
      if (old.length) await supabase.storage.from('avatars').remove(old)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
    setBusy(false)
  }

  const remove = async () => {
    setBusy(true)
    onChange(null)
    const old = await others()
    if (old.length) await supabase.storage.from('avatars').remove(old)
    setBusy(false)
  }

  return (
    <div className="flex items-center gap-4">
      <Avatar name={name || '?'} url={value} size={56} />
      <div className="flex flex-wrap items-center gap-2">
        <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-hairline-strong bg-surface px-2.5 text-xs font-medium text-ink transition-colors hover:bg-surface-hover">
          {busy ? <Spinner className="h-3 w-3" /> : null}
          {value ? 'Change picture' : 'Upload picture'}
          <input type="file" accept="image/png,image/jpeg,image/webp" onChange={upload} disabled={busy} hidden />
        </label>
        {value && (
          <Button type="button" size="sm" variant="ghost" onClick={remove} disabled={busy}>
            Remove
          </Button>
        )}
      </div>
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  )
}
