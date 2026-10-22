import { useState } from 'react'
import { supabase } from '../lib/supabase'
import { Mascot } from './TopBar'
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
            'press flex min-h-16 flex-col items-center justify-center rounded-2xl border-2 px-2 py-2 text-center text-[15px] font-extrabold [--lift:3px]',
            value === v ? 'border-brand-border bg-brand-soft text-brand [--edge:var(--brand-border)]' : 'border-hairline bg-surface text-ink [--edge:var(--border)] hover:bg-surface-hover',
          )}
        >
          {label}
          {sub && <span className={cx('mt-0.5 text-[11px] font-semibold leading-tight', value === v ? 'text-brand' : 'text-ink-muted')}>{sub}</span>}
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
        <p className="text-[15px] font-bold text-ink">Are you more comfortable labelling speech or non-speech sounds?</p>
        <Choice name="Sounds" value={p.sound} onChange={(sound) => change({ sound })}
          options={[['speech', 'Speech', 'whispers, talking, moans'], ['nonspeech', 'Non-speech', 'tapping, crinkles, brushing'], ['both', 'Both']]} />
      </div>
      <div className="space-y-2">
        <p className="text-[15px] font-bold text-ink">Are you more comfortable labelling SFW, NSFW or both kinds of clips?</p>
        <Choice name="Clips" value={p.content} onChange={(content) => change({ content })}
          options={[['sfw', 'SFW'], ['nsfw', 'NSFW'], ['both', 'Both']]} />
      </div>
      {p.content && p.content !== 'sfw' && (
        <div className="space-y-2">
          <p className="text-[15px] font-bold text-ink">In NSFW clips, whose voices do you prefer?</p>
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
      <div className="hp-panel w-full max-w-md rounded-3xl border-2 border-hairline bg-surface-raised p-6">
        <div className="mb-3 flex justify-center"><span className="critter-bob overflow-hidden rounded-2xl"><Mascot size={64} /></span></div>
        <h2 id="pref-title" className="text-center text-2xl font-black tracking-tight text-ink">What would you like to label?</h2>
        <p className="mt-1 mb-5 text-center text-sm font-semibold text-ink-secondary">We’ll only send you clips you’re comfortable with. You can change this any time in Settings.</p>
        <PreferenceFields initial={initial} onSaved={onSaved} saveLabel="Start labelling" />
      </div>
    </div>
  )
}
