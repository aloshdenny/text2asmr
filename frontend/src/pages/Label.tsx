import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'
import Carousel from '../components/Carousel'
import HeadphonesPrompt, { headphonesSeen } from '../components/HeadphonesPrompt'
import Interlude, { type InterludeKind } from '../components/Interlude'
import { AppLayout } from '../components/Layouts'
import PreferencesPrompt, { prefsComplete, type Prefs } from '../components/Preferences'
import TimelineLabeler from '../components/TimelineLabeler'
import PixelCritter from '../components/PixelCritter'
import { Button, ButtonLink, Card, EmptyState, Icon, Input, Kbd, Loading, Spinner, cx } from '../components/ui'
import { useAuth } from '../lib/auth'
import { KEYS, loadMenu, supabase, type Option } from '../lib/supabase'

type Clip = { clip_id: string; audio_path: string; duration_s: number }
type Phase = 'loading' | 'ready' | 'interlude' | 'empty' | 'error'
type Due = { kind: InterludeKind; audio_a: string; audio_b: string | null }
const CHEERS = ['Nice ears!', 'Got it!', 'Lovely, thanks!', 'Sharp listening!', 'That helps a lot!', 'Tingly work!']
type State = { mode: 'tags' | 'timeline'; gold_done: number; gold_total: number; gold_votes: number; sound_pref: Prefs['sound']; content_pref: Prefs['content']; voice_pref: Prefs['voice']; beta: boolean }

/** The classic labeller: tick every sound you hear in the clip. */
function TagLabeler() {
  const [menu, setMenu] = useState<Option[]>([])
  const [clip, setClip] = useState<Clip | null>(null)
  const [src, setSrc] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('loading')
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [other, setOther] = useState('')
  const [started, setStarted] = useState(headphonesSeen) // browsers only play sound after a click: the first clip waits for one
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState({ at: 0, dur: 0 })
  const [hint, setHint] = useState<Option | null>(null)
  const [cheer, setCheer] = useState<{ n: number; text: string } | null>(null) // a quick "nice!" after each answer
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState('')
  const [intro, setIntro] = useState(() => !headphonesSeen()) // the headphones prompt, once per visit
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
    if (picked.size === 0) return flash('Pick at least one label. Not sure? Tick “something else” and say what you hear.')
    if (listenedMs() < 1500) return flash('Listen to the clip first.')
    setBusy(true)
    const { error } = await supabase.rpc('submit_label', {
      p_clip: clip.clip_id,
      p_labels: [...picked],
      p_other: other.trim() || null,
      p_unsure: false,
      p_listen_ms: listenedMs(),
    })
    setBusy(false)
    if (error) return flash(error.message)
    const n = Date.now()
    setCheer({ n, text: CHEERS[n % CHEERS.length] })
    window.setTimeout(() => setCheer((c) => (c?.n === n ? null : c)), 1800)
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
    else if (e.key === 'Escape') setPicked(new Set())
  })
  useEffect(() => {
    const f = (e: KeyboardEvent) => onKey(e)
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  }, [])

  const groups = (['Voice', 'Triggers', 'NSFW', 'Other'] as const).map((g) => [g, menu.filter((o) => o.grp === g)] as const)
  const keyOf = (o: Option) => KEYS[menu.indexOf(o)] ?? ''
  const pct = time.dur ? Math.min(100, (time.at / time.dur) * 100) : 0

  return (
    <div className="pb-28">
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
              className={cx('press flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-brand text-white [--edge:var(--brand-dark)] disabled:opacity-50', playing && 'play-ring')}
            >
              {phase === 'loading' ? (
                <Spinner className="text-white" />
              ) : playing ? (
                <svg viewBox="0 0 24 24" className="h-6 w-6 fill-current" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1.5" /><rect x="14" y="5" width="4" height="14" rx="1.5" /></svg>
              ) : (
                <svg viewBox="0 0 24 24" className="ml-0.5 h-6 w-6 fill-current" aria-hidden><path d="M8 5.5v13l11-6.5z" /></svg>
              )}
            </button>
            <div className="min-w-0 flex-1">
              <div className="mb-2 flex items-baseline justify-between gap-3 max-sm:flex-col max-sm:gap-0.5">
                <span className="truncate text-base font-extrabold text-ink">
                  {phase === 'loading' ? 'Loading the next clip…' : started ? 'What do you hear?' : 'Press play to start listening'}
                </span>
                <span className="shrink-0 font-mono text-xs font-semibold tabular-nums text-ink-muted">
                  {time.at.toFixed(1)} / {(time.dur || clip?.duration_s || 0).toFixed(1)}s
                </span>
              </div>
              <div
                className="h-3 cursor-pointer overflow-hidden rounded-full bg-surface-hover"
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
              <Icon className="h-5 w-5"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></Icon>
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
            <p className="mb-4 text-[15px] font-semibold text-ink-secondary">Tick everything you can hear. Most clips have more than one sound.</p>
            <div className="space-y-4">
              {groups.map(([g, opts]) => opts.length > 0 && (
                <Carousel key={g} title={g} options={opts} count={(k) => (picked.has(k) ? 1 : 0)} onPick={(o) => toggle(o.key)} keyOf={keyOf} onHint={setHint} />
              ))}
            </div>
            {picked.has('something else') && (
              <Input
                ref={otherRef}
                className="mt-5"
                placeholder="What do you hear? (a wooden comb, a lighter, rain…)"
                maxLength={200}
                value={other}
                onChange={(e) => setOther(e.target.value)}
              />
            )}
            <p className="mt-5 min-h-5 border-t-2 border-hairline pt-3 text-sm text-ink-muted">
              {hint ? <><span className="font-bold text-ink">{hint.title ?? hint.key}:</span> {hint.hint}</> : 'Hover a label to see what it covers.'}
            </p>
          </Card>

          {/* the answer bar, like a lesson's check bar: a cheer after each answer, Submit on the right */}
          <div className="fixed inset-x-0 bottom-0 z-20 border-t-2 border-hairline bg-surface pb-[env(safe-area-inset-bottom)]">
            <div className="mx-auto flex h-20 max-w-5xl items-center gap-3 px-4 sm:px-6">
              <div className="min-w-0 flex-1">
                {cheer ? (
                  <div key={cheer.n} className="cheer-in flex items-center gap-3">
                    <span className="overflow-hidden rounded-xl"><PixelCritter seed={`cheer-${cheer.n}`} size={40} sleepy={false} /></span>
                    <span className="text-lg font-black text-ok">{cheer.text}</span>
                  </div>
                ) : (
                  <p className="hidden text-xs font-semibold text-ink-muted sm:block">
                    <Kbd>Space</Kbd> play · <Kbd>R</Kbd> replay · keys on each label · <Kbd>↵</Kbd> submit
                  </p>
                )}
              </div>
              <Button variant="primary" size="lg" className="min-w-36" onClick={submit} disabled={busy || phase !== 'ready'}>Submit</Button>
            </div>
          </div>
        </>
      )}
      {toast && (
        <div role="status" className="fixed bottom-24 left-1/2 z-30 -translate-x-1/2 rounded-2xl bg-ink px-4 py-2.5 text-sm font-bold text-ink-inverse">
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
    </div>
  )
}

/** Preferences first (asked once), then the classic labeller or the timeline one (when the gold set is complete, or for
 *  someone opted into the beta: /label?beta=1 turns it on for that account, ?beta=0 off; nothing in the UI mentions it).
 *  Guests are never asked: they get SFW clips only (the database keeps them there) until they sign up. */
export default function Label() {
  const { profile } = useAuth()
  const [state, setState] = useState<State | null>(null)
  const [error, setError] = useState('')
  const load = useCallback(async () => {
    const flag = new URLSearchParams(window.location.search).get('beta')
    if (flag === '1' || flag === '0') {
      await supabase.rpc('set_preferences', { p_beta: flag === '1' })
      window.history.replaceState(null, '', window.location.pathname)
    }
    const { data, error } = await supabase.rpc('labelling_state')
    if (error) return setError(error.message)
    setState((data as State[])[0])
  }, [])
  useEffect(() => { load() }, [load])
  const leaveBeta = async () => {
    await supabase.rpc('set_preferences', { p_beta: false })
    load()
  }
  if (error) return <AppLayout width="narrow"><EmptyState title="Something went wrong" description={error} action={<Button onClick={load}>Try again</Button>} /></AppLayout>
  if (!state) return <AppLayout width="narrow"><Loading /></AppLayout>
  const prefs: Prefs = { sound: state.sound_pref, content: state.content_pref, voice: state.voice_pref }
  if (!profile?.guest && !prefsComplete(prefs)) {
    return <AppLayout width="narrow"><PreferencesPrompt initial={prefs} onSaved={() => load()} /></AppLayout>
  }
  return (
    <AppLayout>
      {state.mode === 'timeline' ? <TimelineLabeler beta={state.beta} onLeaveBeta={leaveBeta} /> : <TagLabeler />}
    </AppLayout>
  )
}
