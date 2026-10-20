import { useState } from 'react'
import { supabase } from '../lib/supabase'
import { Alert, Button, cx } from './ui'

/** What a labeller is comfortable labelling. Private: only the owner reads or changes it (set_preferences /
 *  labelling_state), and serving honours it (SFW / NSFW and the voice strictly; speech or not as an order). */
export type Prefs = { sound: Sound | null; content: Content | null; voice: Voice | null }
type Sound = 'speech' | 'nonspeech' | 'both'
type Content = 'sfw' | 'nsfw' | 'both'
type Voice = 'female' | 'male' | 'both'

/** Every question answered (the voice only matters once NSFW clips are in). */
export const prefsComplete = (p: Prefs) => Boolean(p.sound && p.content && (p.content === 'sfw' || p.voice))

function Choice<T extends string>({ name, value, options, onChange }: {
  name: string
  value: T | null
  options: [T, string, string?][]
  onChange: (v: T) => void
}) {
  return (
    <div role="radiogroup" aria-label={name} className="grid grid-cols-3 gap-2">
      {options.map(([v, label, sub]) => (
        <button
          key={v}
          type="button"
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={cx(
            'flex min-h-14 flex-col items-center justify-center rounded-xl border px-2 py-2 text-center text-sm transition-colors',
            value === v ? 'border-brand-border bg-brand-soft font-medium text-brand' : 'border-hairline bg-surface text-ink hover:bg-surface-hover',
          )}
        >
          {label}
          {sub && <span className={cx('mt-0.5 text-[11px] leading-tight', value === v ? 'text-brand' : 'text-ink-muted')}>{sub}</span>}
        </button>
      ))}
    </div>
  )
}

/** The three questions, saved together. */
export function PreferenceFields({ initial, onSaved, saveLabel = 'Save' }: { initial: Prefs; onSaved: (p: Prefs) => void; saveLabel?: string }) {
  const [p, setP] = useState<Prefs>(initial)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const change = (part: Partial<Prefs>) => {
    setSaved(false)
    setP((x) => ({ ...x, ...part }))
  }
  const save = async () => {
    if (!prefsComplete(p)) return
    setBusy(true)
    setError('')
    const { error } = await supabase.rpc('set_preferences', { p_sound: p.sound, p_content: p.content, p_voice: p.content === 'sfw' ? null : p.voice })
    setBusy(false)
    if (error) return setError(error.message)
    setSaved(true)
    onSaved(p)
  }
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <p className="text-sm font-medium text-ink">Are you more comfortable labelling speech or non-speech sounds?</p>
        <Choice name="Sounds" value={p.sound} onChange={(sound) => change({ sound })}
          options={[['speech', 'Speech', 'whispers, talking, moans'], ['nonspeech', 'Non-speech', 'tapping, crinkles, brushing'], ['both', 'Both']]} />
      </div>
      <div className="space-y-2">
        <p className="text-sm font-medium text-ink">Are you more comfortable labelling SFW, NSFW or both kinds of clips?</p>
        <Choice name="Clips" value={p.content} onChange={(content) => change({ content })}
          options={[['sfw', 'SFW'], ['nsfw', 'NSFW'], ['both', 'Both']]} />
      </div>
      {p.content && p.content !== 'sfw' && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-ink">In NSFW clips, whose voices do you prefer?</p>
          <Choice name="Voices" value={p.voice} onChange={(voice) => change({ voice })}
            options={[['female', 'Female'], ['male', 'Male'], ['both', 'Both']]} />
        </div>
      )}
      {error && <Alert>{error}</Alert>}
      <div className="flex items-center gap-3">
        <Button variant="primary" disabled={busy || !prefsComplete(p)} onClick={save}>{busy ? 'Saving…' : saveLabel}</Button>
        {saved && <span className="text-sm text-ink-muted">Saved</span>}
      </div>
    </div>
  )
}

/** Asked once of everyone who labels, before the first clip. */
export default function PreferencesPrompt({ initial, onSaved }: { initial: Prefs; onSaved: (p: Prefs) => void }) {
  return (
    <div role="dialog" aria-modal="true" aria-labelledby="pref-title"
      className="hp-backdrop fixed inset-0 z-40 flex items-center justify-center overflow-y-auto bg-black/45 px-4 py-6 backdrop-blur-[3px]">
      <div className="hp-panel w-full max-w-md rounded-2xl border border-hairline bg-surface-raised p-6 shadow-[0_28px_70px_-24px_rgb(0_0_0/0.6)]">
        <h2 id="pref-title" className="text-lg font-semibold tracking-tight text-ink">What would you like to label?</h2>
        <p className="mt-1 mb-5 text-sm text-ink-secondary">We’ll only send you clips you’re comfortable with. You can change this any time in Settings.</p>
        <PreferenceFields initial={initial} onSaved={onSaved} saveLabel="Start labelling" />
      </div>
    </div>
  )
}
