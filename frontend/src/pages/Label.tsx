import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'
import HeadphonesPrompt from '../components/HeadphonesPrompt'
import Interlude, { type InterludeKind } from '../components/Interlude'
import { AppLayout } from '../components/Layouts'
import { Button, ButtonLink, Card, EmptyState, Icon, Kbd, Spinner, cx } from '../components/ui'
import { KEYS, loadMenu, supabase, type Option } from '../lib/supabase'

type Clip = { clip_id: string; audio_path: string; duration_s: number }
type Phase = 'loading' | 'ready' | 'interlude' | 'empty' | 'error'
type Due = { kind: InterludeKind; audio_a: string; audio_b: string | null }
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
  const [intro, setIntro] = useState(true) // the headphones prompt, once per visit
  const [inter, setInter] = useState<{ kind: InterludeKind; a: string; b: string | null } | null>(null)
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
    // every 8-17 labels the server has a quick break (Real or AI / Which is hotter) instead; the clip stays assigned
    const [due, clipRes] = await Promise.all([supabase.rpc('next_interlude'), supabase.rpc('next_clip')])
    const it = (due.data as Due[] | null)?.[0]
    if (it) {
      const sign = (path: string) => supabase.storage.from('clips').createSignedUrl(path, 900)
      const [sa, sb] = await Promise.all([sign(it.audio_a), it.audio_b ? sign(it.audio_b) : null])
      if (sa.data && (!it.audio_b || sb?.data)) {
        setInter({ kind: it.kind, a: sa.data.signedUrl, b: sb?.data?.signedUrl ?? null })
        return setPhase('interlude')
      }
      await supabase.rpc('skip_interlude') // its audio would not load: carry on labelling
    }
    const { data, error } = clipRes
    if (error) {
      setError(error.message)
      return setPhase('error')
    }
    const c = (data as Clip[])[0]
    if (!c) {
      setClip(null)
      setSrc(null)
      return setPhase('empty')
    }
    const { data: signed, error: e2 } = await supabase.storage.from('clips').createSignedUrl(c.audio_path, 900)
    if (e2 || !signed) {
      setError(e2?.message ?? 'Could not load the audio')
      return setPhase('error')
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
    if (picked.size === 0 && !unsure) return flash('Pick at least one label, or “Can’t tell”.')
    if (listenedMs() < 1500) return flash('Listen to the clip first.')
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
    setTotals((data as { total: number; today: number }[])[0])
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
    if (intro || phase === 'interlude') return // the headphones prompt / the interlude has the keyboard
    const t = e.target as HTMLElement
    if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) {
      if (e.key === 'Enter') {
        e.preventDefault()
        submit()
      } else if (e.key === 'Escape') t.blur()
      return
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return
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

  const groups = (['Voice', 'Triggers', 'Other'] as const).map((g) => [g, menu.filter((o) => o.grp === g)] as const)
  const keyOf = (o: Option) => KEYS[menu.indexOf(o)] ?? ''
  const pct = time.dur ? Math.min(100, (time.at / time.dur) * 100) : 0

  return (
    <AppLayout width="narrow">
      <div className="mb-5 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 text-sm text-ink-muted">
          <div className="h-1.5 w-28 overflow-hidden rounded-full bg-surface-hover" aria-label={`${done} of ${SESSION_GOAL} this session`}>
            <div className="h-full rounded-full bg-brand transition-[width] duration-300" style={{ width: `${Math.min(100, (done / SESSION_GOAL) * 100)}%` }} />
          </div>
          <span>
            <span className="font-medium text-ink tabular-nums">{done}</span> this session
            {totals && <span className="hidden sm:inline"> · {totals.today} today · {totals.total.toLocaleString()} all time</span>}
          </span>
        </div>
      </div>

      {phase === 'empty' && (
        <EmptyState title="You’ve heard every clip" description="Thank you, that’s the whole queue. New clips arrive regularly." action={<ButtonLink to="/dashboard">Back to your dashboard</ButtonLink>} />
      )}
      {phase === 'error' && <EmptyState title="Something went wrong" description={error} action={<Button onClick={next}>Try again</Button>} />}
      {phase === 'interlude' && inter && (
        <Interlude
          kind={inter.kind}
          srcA={inter.a}
          srcB={inter.b}
          autoPlay={!intro}
          onError={flash}
          onDone={() => {
            setInter(null)
            next()
          }}
        />
      )}

      {(phase === 'ready' || phase === 'loading') && (
        <>
          <Card className="flex items-center gap-4">
            <button
              onClick={play}
              disabled={phase !== 'ready'}
              aria-label={playing ? 'Pause' : 'Play'}
              className={cx('flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-brand text-white transition-colors hover:bg-brand-dark disabled:opacity-50', playing && 'play-ring')}
            >
              {phase === 'loading' ? (
                <Spinner className="text-white" />
              ) : playing ? (
                <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
              ) : (
                <svg viewBox="0 0 24 24" className="ml-0.5 h-5 w-5 fill-current" aria-hidden><path d="M8 5.5v13l11-6.5z" /></svg>
              )}
            </button>
            <div className="min-w-0 flex-1">
              <div className="mb-2 flex items-baseline justify-between gap-3 max-sm:flex-col max-sm:gap-0.5">
                <span className="truncate text-sm font-medium text-ink">
                  {phase === 'loading' ? 'Loading the next clip…' : started ? 'What do you hear?' : 'Press play to start listening'}
                </span>
                <span className="shrink-0 font-mono text-xs tabular-nums text-ink-muted">
                  {time.at.toFixed(1)} / {(time.dur || clip?.duration_s || 0).toFixed(1)}s
                </span>
              </div>
              <div
                className="h-1.5 cursor-pointer overflow-hidden rounded-full bg-surface-hover"
                onClick={(e) => {
                  const a = audio.current
                  if (!a || !a.duration) return
                  const r = e.currentTarget.getBoundingClientRect()
                  a.currentTime = ((e.clientX - r.left) / r.width) * a.duration
                }}
              >
                <div className="h-full rounded-full bg-brand" style={{ width: `${pct}%` }} />
              </div>
            </div>
            <Button size="sm" variant="ghost" onClick={replay} disabled={phase !== 'ready'} title="Replay (R)" aria-label="Replay">
              <Icon className="sm:hidden"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></Icon>
              <span className="max-sm:hidden">Replay</span> <Kbd className="max-sm:hidden">R</Kbd>
            </Button>
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
          </Card>

          <Card className="mt-3">
            <p className="mb-4 text-sm text-ink-muted">Tick everything you can hear. Most clips have more than one sound.</p>
            <div className="space-y-4">
              {groups.map(([g, opts]) => (
                <div key={g}>
                  <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{g}</h3>
                  <div className="flex flex-wrap gap-1.5">
                    {opts.map((o) => {
                      const on = picked.has(o.key)
                      return (
                        <button
                          key={o.key}
                          aria-pressed={on}
                          title={o.hint}
                          onMouseEnter={() => setHint(o)}
                          onFocus={() => setHint(o)}
                          onClick={() => toggle(o.key)}
                          className={cx(
                            'inline-flex h-8 items-center gap-2 rounded-lg border pl-1.5 pr-2.5 text-[13px] transition-colors max-sm:pl-2.5',
                            on ? 'border-brand-border bg-brand-soft font-medium text-brand' : 'border-hairline bg-surface text-ink hover:bg-surface-hover',
                          )}
                        >
                          <Kbd className={cx('max-sm:hidden', on && 'border-brand-border bg-transparent text-brand')}>{keyOf(o)}</Kbd>
                          {o.key}
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
            {picked.has('something else') && (
              <input
                ref={otherRef}
                className="mt-4 h-9 w-full rounded-lg border border-hairline-strong bg-surface px-3 text-sm text-ink placeholder:text-ink-muted hover:border-brand-border focus:border-brand focus:outline-none"
                placeholder="What do you hear? (a wooden comb, a lighter, rain…)"
                maxLength={200}
                value={other}
                onChange={(e) => setOther(e.target.value)}
              />
            )}
            <p className="mt-4 min-h-5 border-t border-hairline pt-3 text-xs text-ink-muted">
              {hint ? <><span className="font-medium text-ink">{hint.key}:</span> {hint.hint}</> : 'Hover a label to see what it covers.'}
            </p>
          </Card>

          <div className="mt-3 flex items-center gap-2">
            <Button variant={unsure ? 'secondary' : 'ghost'} aria-pressed={unsure} onClick={() => setUnsure((u) => !u)} className={cx(unsure && 'border-brand-border bg-brand-soft text-brand')}>
              Can’t tell <Kbd className="max-sm:hidden">U</Kbd>
            </Button>
            <Button variant="ghost" onClick={skip} disabled={busy || phase !== 'ready'}>Skip</Button>
            <span className="flex-1" />
            <Button variant="primary" onClick={submit} disabled={busy || phase !== 'ready'}>
              Submit <Kbd className="border-white/30 bg-transparent text-white/80 max-sm:hidden">↵</Kbd>
            </Button>
          </div>
          <p className="mt-4 hidden text-xs text-ink-muted sm:block">
            <Kbd>1</Kbd> to <Kbd>0</Kbd> and <Kbd>A</Kbd> to <Kbd>J</Kbd> toggle labels · <Kbd>Space</Kbd> play / pause · <Kbd>R</Kbd> replay · <Kbd>U</Kbd> can’t tell · <Kbd>↵</Kbd> submit · <Kbd>Esc</Kbd> clear
          </p>
        </>
      )}
      {toast && (
        <div role="status" className="fixed bottom-6 left-1/2 z-30 -translate-x-1/2 rounded-lg bg-ink px-3.5 py-2 text-sm font-medium text-ink-inverse shadow-lg">
          {toast}
        </div>
      )}
      {intro && (
        <HeadphonesPrompt
          onReady={() => {
            setStarted(true)
            audio.current?.play().catch(() => {})
          }}
          onClosed={() => setIntro(false)}
        />
      )}
    </AppLayout>
  )
}
