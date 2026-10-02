import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import GuidePanel from '../components/GuidePanel'
import { KEYS, loadMenu, supabase, type Option } from '../lib/supabase'

type Clip = { clip_id: string; audio_path: string; duration_s: number }
type Phase = 'loading' | 'ready' | 'empty' | 'error'
const SESSION_GOAL = 25

export default function Label() {
  const [menu, setMenu] = useState<Option[]>([])
  const [clip, setClip] = useState<Clip | null>(null)
  const [src, setSrc] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('loading')
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [other, setOther] = useState('')
  const [unsure, setUnsure] = useState(false)
  const [started, setStarted] = useState(false) // browsers only play sound after a click: the first clip waits for one
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState({ at: 0, dur: 0 })
  const [hint, setHint] = useState<Option | null>(null)
  const [done, setDone] = useState(0)
  const [totals, setTotals] = useState<{ total: number; today: number } | null>(null)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState('')
  const [guideOpen, setGuideOpen] = useState(false)
  const audio = useRef<HTMLAudioElement>(null)
  const otherRef = useRef<HTMLInputElement>(null)

  const flash = (m: string) => {
    setToast(m)
    window.setTimeout(() => setToast((t) => (t === m ? '' : t)), 2600)
  }

  const next = useCallback(async () => {
    audio.current?.pause()
    setPhase('loading')
    setPicked(new Set())
    setOther('')
    setUnsure(false)
    setTime({ at: 0, dur: 0 })
    const { data, error } = await supabase.rpc('next_clip')
    if (error) {
      setError(error.message)
      setPhase('error')
      return
    }
    const c = (data as Clip[])[0]
    if (!c) {
      setClip(null)
      setSrc(null)
      setPhase('empty')
      return
    }
    const { data: signed, error: e2 } = await supabase.storage.from('clips').createSignedUrl(c.audio_path, 900)
    if (e2 || !signed) {
      setError(e2?.message ?? 'Could not load the audio')
      setPhase('error')
      return
    }
    setClip(c)
    setSrc(signed.signedUrl)
    setPhase('ready')
  }, [])

  useEffect(() => {
    loadMenu().then(setMenu, (e) => setError(String(e.message ?? e)))
    next()
  }, [next])

  useEffect(() => {
    if (src && started) audio.current?.play().catch(() => setPlaying(false))
  }, [src, started])

  const listenedMs = () => {
    const a = audio.current
    if (!a) return 0
    let t = 0
    for (let i = 0; i < a.played.length; i++) t += a.played.end(i) - a.played.start(i)
    return Math.round(t * 1000)
  }

  const toggle = (key: string) => {
    setPicked((p) => {
      const n = new Set(p)
      if (n.has(key)) n.delete(key)
      else n.add(key)
      return n
    })
    if (key === 'something else' && !picked.has(key)) window.setTimeout(() => otherRef.current?.focus(), 0)
  }

  const play = () => {
    const a = audio.current
    if (!a) return
    if (!started) setStarted(true)
    if (a.paused) a.play().catch(() => {})
    else a.pause()
  }
  const replay = () => {
    const a = audio.current
    if (!a) return
    a.currentTime = 0
    a.play().catch(() => {})
  }

  const submit = async () => {
    if (!clip || busy || phase !== 'ready') return
    if (picked.size === 0 && !unsure) return flash('Pick at least one label, or “Can’t tell”')
    if (listenedMs() < 1500) return flash('Listen to the clip first')
    setBusy(true)
    const { data, error } = await supabase.rpc('submit_label', {
      p_clip: clip.clip_id,
      p_labels: [...picked],
      p_other: other.trim() || null,
      p_unsure: unsure,
      p_listen_ms: listenedMs(),
    })
    setBusy(false)
    if (error) return flash(error.message)
    const t = (data as { total: number; today: number }[])[0]
    setTotals(t)
    setDone((d) => d + 1)
    next()
  }

  const skip = async () => {
    if (!clip || busy) return
    setBusy(true)
    await supabase.rpc('skip_clip', { p_clip: clip.clip_id })
    setBusy(false)
    next()
  }

  const onKey = useEffectEvent((e: KeyboardEvent) => {
    const t = e.target as HTMLElement
    if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) {
      if (e.key === 'Enter') {
        e.preventDefault()
        submit()
      } else if (e.key === 'Escape') t.blur()
      return
    }
    if (e.metaKey || e.ctrlKey || e.altKey || guideOpen) return
    const k = e.key.toLowerCase()
    const i = KEYS.indexOf(k)
    if (i >= 0 && i < menu.length) {
      e.preventDefault()
      toggle(menu[i].key)
    } else if (e.key === ' ') {
      e.preventDefault()
      play()
    } else if (e.key === 'Enter') {
      e.preventDefault()
      submit()
    } else if (k === 'r') replay()
    else if (k === 'u') setUnsure((u) => !u)
    else if (e.key === 'Escape') {
      setPicked(new Set())
      setUnsure(false)
    }
  })
  useEffect(() => {
    const f = (e: KeyboardEvent) => onKey(e)
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  }, [])

  const groups: [string, Option[]][] = (['Voice', 'Triggers', 'Other'] as const).map((g) => [g, menu.filter((o) => o.grp === g)])
  const keyOf = (o: Option) => KEYS[menu.indexOf(o)] ?? ''
  const pct = time.dur ? Math.min(100, (time.at / time.dur) * 100) : 0

  return (
    <main className="label-page">
      <div className="label-top">
        <div className="session">
          <div className="session-bar" aria-label={`${done} of ${SESSION_GOAL} this session`}>
            <span style={{ width: `${Math.min(100, (done / SESSION_GOAL) * 100)}%` }} />
          </div>
          <span>
            <b>{done}</b> this session{totals ? ` · ${totals.today} today · ${totals.total} all time` : ''}
          </span>
        </div>
        <button className="btn ghost small" onClick={() => setGuideOpen(true)}>
          Sound guide
        </button>
      </div>

      {phase === 'empty' && (
        <section className="card center-card">
          <h2>All caught up</h2>
          <p className="muted">Every clip has enough listeners for now. New clips arrive regularly — check back soon.</p>
          <Link className="btn" to="/leaderboard">See the leaderboard</Link>
        </section>
      )}
      {phase === 'error' && (
        <section className="card center-card">
          <h2>Something went wrong</h2>
          <p className="muted">{error}</p>
          <button className="btn" onClick={next}>Try again</button>
        </section>
      )}

      {(phase === 'ready' || phase === 'loading') && (
        <>
          <section className="card player">
            <button className={`play ${playing ? 'on' : ''}`} onClick={play} disabled={phase !== 'ready'} aria-label={playing ? 'Pause' : 'Play'}>
              {playing ? (
                <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
              ) : (
                <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden><path d="M8 5.5v13l11-6.5z" /></svg>
              )}
            </button>
            <div className="track">
              <div className="track-head">
                <span>{phase === 'loading' ? 'Loading the next clip…' : started ? 'What do you hear?' : 'Press play to start listening'}</span>
                <span className="mono muted">
                  {time.at.toFixed(1)} / {(time.dur || clip?.duration_s || 0).toFixed(1)} s
                </span>
              </div>
              <div
                className="progress"
                onClick={(e) => {
                  const a = audio.current
                  if (!a || !a.duration) return
                  const r = e.currentTarget.getBoundingClientRect()
                  a.currentTime = ((e.clientX - r.left) / r.width) * a.duration
                }}
              >
                <span style={{ width: `${pct}%` }} />
              </div>
            </div>
            <button className="btn ghost small" onClick={replay} disabled={phase !== 'ready'} title="Replay (R)">
              Replay <kbd>R</kbd>
            </button>
            {src && (
              <audio
                ref={audio}
                src={src}
                preload="auto"
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
                onTimeUpdate={(e) => setTime({ at: e.currentTarget.currentTime, dur: e.currentTarget.duration || 0 })}
                onLoadedMetadata={(e) => setTime({ at: 0, dur: e.currentTarget.duration || 0 })}
              />
            )}
          </section>

          <section className="card labels">
            <p className="muted small-print">Tick everything you can hear — most clips have more than one sound.</p>
            {groups.map(([g, opts]) => (
              <div key={g} className="group">
                <h3>{g}</h3>
                <div className="chips">
                  {opts.map((o) => (
                    <button
                      key={o.key}
                      className={`chip ${picked.has(o.key) ? 'on' : ''}`}
                      aria-pressed={picked.has(o.key)}
                      title={o.hint}
                      onMouseEnter={() => setHint(o)}
                      onFocus={() => setHint(o)}
                      onClick={() => toggle(o.key)}
                    >
                      <kbd>{keyOf(o)}</kbd>
                      {o.key}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {picked.has('something else') && (
              <input
                ref={otherRef}
                className="other-input"
                placeholder="What do you hear? (e.g. a wooden comb, a lighter, rain)"
                maxLength={200}
                value={other}
                onChange={(e) => setOther(e.target.value)}
              />
            )}
            <div className="hint-line">{hint ? <><b>{hint.key}:</b> {hint.hint}</> : 'Hover a label to see what it covers.'}</div>
          </section>

          <div className="actions">
            <button className={`btn ghost ${unsure ? 'on' : ''}`} aria-pressed={unsure} onClick={() => setUnsure((u) => !u)} title="Can't tell (U)">
              Can’t tell <kbd>U</kbd>
            </button>
            <button className="btn ghost" onClick={skip} disabled={busy || phase !== 'ready'} title="Pass on this clip">
              Skip
            </button>
            <span className="spacer" />
            <button className="btn primary" onClick={submit} disabled={busy || phase !== 'ready'}>
              Submit <kbd>↵</kbd>
            </button>
          </div>
          <p className="muted keys-help">
            Keys: <kbd>1</kbd>–<kbd>0</kbd>, <kbd>A</kbd>–<kbd>J</kbd> toggle labels · <kbd>Space</kbd> play/pause · <kbd>R</kbd> replay · <kbd>U</kbd> can’t tell · <kbd>Enter</kbd> submit · <kbd>Esc</kbd> clear
          </p>
        </>
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
      {guideOpen && <GuidePanel onClose={() => setGuideOpen(false)} />}
    </main>
  )
}
