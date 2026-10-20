import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import HeadphonesPrompt, { headphonesSeen } from './HeadphonesPrompt'
import Interlude, { type InterludeKind } from './Interlude'
import { Button, ButtonLink, EmptyState, Spinner, cx } from './ui'
import { loadMenu, supabase, type Option } from '../lib/supabase'

/* The timeline labeller (beta): every sound in a clip is a span with a start and an end in milliseconds, dragged on
   the clip's waveform like a clip on a video editor's timeline. Up to six sounds: lanes 0-2 above the waveform
   (0 nearest), lanes 3-5 below (3 nearest). Prev / next revisit the last 24 hours of answers, which can be changed. */

type Span = { id: string; label: string; lane: number; start: number; end: number }
type Entry = {
  clip_id: string
  audio_path: string
  duration: number // ms
  spans: Span[]
  other: string
  submitted: boolean
}
type Recent = { clip_id: string; audio_path: string; duration_s: number; labels: string[]; spans: { label: string; start_ms: number; end_ms: number; lane: number }[] | null; other_text: string | null }
type Due = { kind: InterludeKind; audio_a: string; audio_b: string | null }

const LANE_ORDER = [3, 0, 4, 1, 5, 2] // fill: below 1, above 1, below 2, above 2, below 3, above 3
const MIN_SPAN = 20 // ms
const GROUPS = ['Voice', 'Triggers', 'NSFW', 'Other'] as const
const BARS = 96
let spanSeq = 0
const newId = () => `s${++spanSeq}`
const fmt = (ms: number) => (ms / 1000).toFixed(3)

/** A clip's loudness in BARS buckets (0..1), from its decoded audio. */
async function peaksOf(src: string): Promise<number[]> {
  const buf = await (await fetch(src)).arrayBuffer()
  const ctx = new AudioContext()
  try {
    const audio = await ctx.decodeAudioData(buf)
    const ch = audio.getChannelData(0)
    const step = Math.max(1, Math.floor(ch.length / BARS))
    const out: number[] = []
    for (let b = 0; b < BARS; b++) {
      let m = 0
      for (let i = b * step; i < Math.min(ch.length, (b + 1) * step); i += 8) m = Math.max(m, Math.abs(ch[i]))
      out.push(m)
    }
    const top = Math.max(...out, 1e-6)
    return out.map((v) => Math.max(0.04, v / top))
  } finally {
    ctx.close()
  }
}

export default function TimelineLabeler({ onLeaveBeta, beta }: { onLeaveBeta?: () => void; beta: boolean }) {
  const [menu, setMenu] = useState<Option[]>([])
  const [history, setHistory] = useState<Entry[]>([])
  const [idx, setIdx] = useState(0)
  const [src, setSrc] = useState<string | null>(null)
  const [peaks, setPeaks] = useState<number[]>([])
  const [phase, setPhase] = useState<'loading' | 'ready' | 'interlude' | 'empty' | 'error'>('loading')
  const [error, setError] = useState('')
  const [inter, setInter] = useState<{ kind: InterludeKind; a: string; b: string | null } | null>(null)
  const [intro, setIntro] = useState(() => !headphonesSeen())
  const [started, setStarted] = useState(headphonesSeen)
  const [playing, setPlaying] = useState(false)
  const [at, setAt] = useState(0)
  const [active, setActive] = useState<string | null>(null)
  const [hint, setHint] = useState<Option | null>(null)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState('')
  const [done, setDone] = useState(0)
  const [follow, setFollow] = useState(false) // a clip was just appended: show it
  const audio = useRef<HTMLAudioElement>(null)
  const track = useRef<HTMLDivElement>(null)
  const drag = useRef<{ id: string; edge: 'start' | 'end' | 'move'; x0: number; s0: number; e0: number } | null>(null)
  const raf = useRef(0)

  const entry = history[idx]
  const byKey = (k: string) => menu.find((o) => o.key === k)
  const title = (k: string) => byKey(k)?.title ?? k

  const flash = (m: string) => {
    setToast(m)
    window.setTimeout(() => setToast((t) => (t === m ? '' : t)), 2600)
  }

  const update = (fn: (e: Entry) => Entry) => setHistory((h) => h.map((e, i) => (i === idx ? fn(e) : e)))

  // the next clip (or a quick break): appended to the history and shown
  const fetchNext = useCallback(async () => {
    setPhase('loading')
    const [due, clipRes] = await Promise.all([supabase.rpc('next_interlude'), supabase.rpc('next_clip')])
    const it = (due.data as Due[] | null)?.[0]
    if (it) {
      const sign = (p: string) => supabase.storage.from('clips').createSignedUrl(p, 900)
      const [sa, sb] = await Promise.all([sign(it.audio_a), it.audio_b ? sign(it.audio_b) : null])
      if (sa.data && (!it.audio_b || sb?.data)) {
        setInter({ kind: it.kind, a: sa.data.signedUrl, b: sb?.data?.signedUrl ?? null })
        return setPhase('interlude')
      }
      await supabase.rpc('skip_interlude')
    }
    if (clipRes.error) {
      setError(clipRes.error.message)
      return setPhase('error')
    }
    const c = (clipRes.data as { clip_id: string; audio_path: string; duration_s: number }[])[0]
    if (!c) return setPhase('empty')
    setHistory((h) => [...h.slice(-20), { clip_id: c.clip_id, audio_path: c.audio_path, duration: Math.round((c.duration_s || 6) * 1000), spans: [], other: '', submitted: false }])
    setFollow(true)
    setPhase('ready')
  }, [])

  // the menu, the last 24 hours of answers (for prev), then a clip
  useEffect(() => {
    loadMenu().then(setMenu, (e) => setError(String(e.message ?? e)))
    supabase.rpc('my_recent_answers', { p_limit: 20 }).then(({ data }) => {
      const past = ((data as Recent[] | null) ?? []).reverse().map<Entry>((r) => {
        const duration = Math.round((r.duration_s || 6) * 1000)
        const spans = r.spans?.length
          ? r.spans.map((s) => ({ id: newId(), label: s.label, lane: s.lane, start: s.start_ms, end: s.end_ms }))
          : r.labels.filter((l) => l !== 'something else').slice(0, 6).map((l, i) => ({ id: newId(), label: l, lane: LANE_ORDER[i], start: 0, end: duration }))
        return { clip_id: r.clip_id, audio_path: r.audio_path, duration, spans, other: r.other_text ?? '', submitted: true }
      })
      setHistory(past)
      fetchNext()
    })
  }, [fetchNext])

  useEffect(() => {
    if (!follow) return
    setIdx(history.length - 1)
    setFollow(false)
  }, [follow, history.length])

  // audio and waveform for the entry on screen
  useEffect(() => {
    if (!entry) return
    let gone = false
    setSrc(null)
    setPeaks([])
    setAt(0)
    setActive(null)
    supabase.storage.from('clips').createSignedUrl(entry.audio_path, 900).then(({ data }) => {
      if (gone || !data) return
      setSrc(data.signedUrl)
      peaksOf(data.signedUrl).then((p) => !gone && setPeaks(p), () => {})
    })
    return () => { gone = true }
  }, [entry?.clip_id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (src && started && entry && !entry.submitted) audio.current?.play().catch(() => setPlaying(false))
  }, [src, started]) // eslint-disable-line react-hooks/exhaustive-deps

  // a smooth playhead while playing
  useEffect(() => {
    const tick = () => {
      if (audio.current) setAt(audio.current.currentTime * 1000)
      raf.current = requestAnimationFrame(tick)
    }
    if (playing) raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing])

  const play = () => {
    const a = audio.current
    if (!a) return
    if (!started) setStarted(true)
    if (a.paused) a.play().catch(() => {})
    else a.pause()
  }
  const seek = (ms: number) => {
    const a = audio.current
    if (a) a.currentTime = ms / 1000
    setAt(ms)
  }
  const listenedMs = () => {
    const a = audio.current
    if (!a) return 0
    let t = 0
    for (let i = 0; i < a.played.length; i++) t += a.played.end(i) - a.played.start(i)
    return Math.round(t * 1000)
  }

  // spans
  const addSpan = (o: Option) => {
    if (!entry) return
    setHint(o)
    const used = new Set(entry.spans.map((s) => s.lane))
    const lane = LANE_ORDER.find((l) => !used.has(l))
    if (lane === undefined) return flash('Six sounds per clip: remove one to add another.')
    const id = newId()
    update((e) => ({ ...e, spans: [...e.spans, { id, label: o.key, lane, start: 0, end: e.duration }] }))
    setActive(id)
  }
  const removeSpan = (id: string) => update((e) => ({ ...e, spans: e.spans.filter((s) => s.id !== id) }))
  const setSpan = (id: string, start: number, end: number) =>
    update((e) => ({ ...e, spans: e.spans.map((s) => (s.id === id ? { ...s, start: Math.round(start), end: Math.round(end) } : s)) }))

  const msAt = (clientX: number) => {
    const r = track.current?.getBoundingClientRect()
    if (!r || !entry) return 0
    return Math.min(entry.duration, Math.max(0, ((clientX - r.left) / r.width) * entry.duration))
  }
  const onDown = (e: ReactPointerEvent, s: Span, edge: 'start' | 'end' | 'move') => {
    e.stopPropagation()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    drag.current = { id: s.id, edge, x0: e.clientX, s0: s.start, e0: s.end }
    setActive(s.id)
    setHint(byKey(s.label) ?? null)
  }
  const onMove = (e: ReactPointerEvent) => {
    const d = drag.current
    if (!d || !entry || !track.current) return
    const dx = ((e.clientX - d.x0) / track.current.getBoundingClientRect().width) * entry.duration
    if (d.edge === 'start') {
      const s = Math.min(Math.max(0, d.s0 + dx), d.e0 - MIN_SPAN)
      setSpan(d.id, s, d.e0)
      seek(s)
    } else if (d.edge === 'end') {
      const t = Math.max(Math.min(entry.duration, d.e0 + dx), d.s0 + MIN_SPAN)
      setSpan(d.id, d.s0, t)
      seek(t)
    } else {
      const len = d.e0 - d.s0
      const s = Math.min(Math.max(0, d.s0 + dx), entry.duration - len)
      setSpan(d.id, s, s + len)
    }
  }
  const onUp = () => { drag.current = null }
  const nudge = (s: Span, edge: 'start' | 'end', by: number) => {
    if (!entry) return
    if (edge === 'start') setSpan(s.id, Math.min(Math.max(0, s.start + by), s.end - MIN_SPAN), s.end)
    else setSpan(s.id, s.start, Math.max(Math.min(entry.duration, s.end + by), s.start + MIN_SPAN))
  }

  const submit = async () => {
    if (!entry || busy) return
    if (entry.spans.length === 0 && !entry.other.trim()) return flash('Mark at least one sound, or say what you hear.')
    const spans = entry.spans.map((s) => ({ label: s.label, start_ms: s.start, end_ms: s.end, lane: s.lane }))
    if (entry.submitted) {
      setBusy(true)
      const { error } = await supabase.rpc('edit_answer', { p_clip: entry.clip_id, p_spans: spans, p_other: entry.other.trim() || null })
      setBusy(false)
      return flash(error ? error.message : 'Saved your changes.')
    }
    if (listenedMs() < 1500) return flash('Listen to the clip first.')
    setBusy(true)
    audio.current?.pause()
    const { error } = await supabase.rpc('submit_spans', { p_clip: entry.clip_id, p_spans: spans, p_other: entry.other.trim() || null, p_listen_ms: listenedMs() })
    setBusy(false)
    if (error) return flash(error.message)
    update((e) => ({ ...e, submitted: true }))
    setDone((d) => d + 1)
    fetchNext()
  }

  // Space plays, Enter submits (not while typing)
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (intro || phase !== 'ready' || e.metaKey || e.ctrlKey || e.altKey) return
      const t = e.target as HTMLElement
      if (t instanceof HTMLInputElement) {
        if (e.key === 'Enter') { e.preventDefault(); submit() }
        return
      }
      if (t instanceof HTMLButtonElement && t.getAttribute('role') === 'slider') return
      if (e.key === ' ') { e.preventDefault(); play() }
      else if (e.key === 'Enter' && !(t instanceof HTMLButtonElement)) { e.preventDefault(); submit() }
    }
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  })

  const go = (to: number) => {
    audio.current?.pause()
    setIdx(Math.max(0, Math.min(history.length - 1, to)))
  }

  const lane = (n: number) => {
    const s = entry?.spans.find((x) => x.lane === n)
    return (
      <div key={n} className="relative h-8">
        {!s ? (
          <div className="absolute inset-x-0 inset-y-1 rounded-lg border border-dashed border-hairline" />
        ) : (
          <div
            className={cx('absolute inset-y-0.5 flex touch-none select-none items-center rounded-lg border text-xs font-medium',
              active === s.id ? 'z-10 border-brand bg-brand-soft text-brand' : 'border-brand-border bg-brand-soft/70 text-brand')}
            style={{ left: `${(s.start / entry!.duration) * 100}%`, width: `${((s.end - s.start) / entry!.duration) * 100}%` }}
            onPointerDown={(e) => onDown(e, s, 'move')} onPointerMove={onMove} onPointerUp={onUp}
          >
            <button type="button" role="slider" aria-label={`${title(s.label)} start`} aria-valuemin={0} aria-valuemax={entry!.duration} aria-valuenow={s.start}
              className="absolute -left-2 top-0 flex h-full w-4 cursor-ew-resize items-center justify-center"
              onPointerDown={(e) => onDown(e, s, 'start')} onPointerMove={onMove} onPointerUp={onUp}
              onKeyDown={(e) => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); nudge(s, 'start', (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 100 : 10)) } }}>
              <span className="h-4 w-1 rounded-full bg-brand" />
            </button>
            <span className="min-w-0 flex-1 truncate px-3">{title(s.label)}</span>
            <button type="button" aria-label={`Remove ${title(s.label)}`} className="mr-2 shrink-0 rounded px-1 text-brand/70 hover:text-brand"
              onPointerDown={(e) => e.stopPropagation()} onClick={() => removeSpan(s.id)}>×</button>
            <button type="button" role="slider" aria-label={`${title(s.label)} end`} aria-valuemin={0} aria-valuemax={entry!.duration} aria-valuenow={s.end}
              className="absolute -right-2 top-0 flex h-full w-4 cursor-ew-resize items-center justify-center"
              onPointerDown={(e) => onDown(e, s, 'end')} onPointerMove={onMove} onPointerUp={onUp}
              onKeyDown={(e) => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); nudge(s, 'end', (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 100 : 10)) } }}>
              <span className="h-4 w-1 rounded-full bg-brand" />
            </button>
          </div>
        )}
      </div>
    )
  }

  const act = entry?.spans.find((s) => s.id === active)
  const pct = entry ? Math.min(100, (at / entry.duration) * 100) : 0

  return (
    <>
      <div className="mb-4 flex items-center justify-between gap-3 text-sm text-ink-muted">
        <span><span className="font-medium text-ink tabular-nums">{done}</span> this session · <span className="rounded bg-brand-soft px-1.5 py-0.5 text-xs font-medium text-brand">Timeline beta</span></span>
        {beta && onLeaveBeta && <button className="text-xs hover:text-ink" onClick={onLeaveBeta}>Back to the classic labeller</button>}
      </div>

      {phase === 'empty' && <EmptyState title="You’ve heard every clip" description="That’s the whole queue for now." action={<ButtonLink to="/dashboard">Back to your dashboard</ButtonLink>} />}
      {phase === 'error' && <EmptyState title="Something went wrong" description={error} action={<Button onClick={fetchNext}>Try again</Button>} />}
      {phase === 'interlude' && inter && (
        <Interlude kind={inter.kind} srcA={inter.a} srcB={inter.b} autoPlay={!intro} onError={flash} onDone={() => { setInter(null); fetchNext() }} />
      )}

      {(phase === 'ready' || phase === 'loading') && entry && (
        <div className="space-y-3">
          <div className="rounded-2xl border border-hairline bg-surface p-4 sm:p-5">
            <div className="mb-3 flex items-center justify-between text-xs text-ink-muted">
              <span>{entry.submitted ? 'Your answer · you can still change it' : 'What do you hear, and when?'}</span>
              <span className="font-mono tabular-nums">{fmt(at)} / {fmt(entry.duration)} s</span>
            </div>
            <div className="flex items-center gap-3 sm:gap-4">
              <button onClick={play} disabled={!src} aria-label={playing ? 'Pause' : 'Play'}
                className={cx('flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-brand text-white disabled:opacity-50', playing && 'play-ring')}>
                {!src ? <Spinner className="text-white" /> : playing
                  ? <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                  : <svg viewBox="0 0 24 24" className="ml-0.5 h-5 w-5 fill-current" aria-hidden><path d="M8 5.5v13l11-6.5z" /></svg>}
              </button>
              <div className="min-w-0 flex-1 select-none" ref={track}>
                {[2, 1, 0].map(lane)}
                <div className="relative my-1 h-14 cursor-pointer rounded-lg bg-surface-muted" onPointerDown={(e) => seek(msAt(e.clientX))} aria-label="Seek">
                  <div className="absolute inset-0 flex items-center gap-px px-1">
                    {(peaks.length ? peaks : Array(BARS).fill(0.06)).map((v, i) => (
                      <div key={i} className={cx('flex-1 rounded-full', (i + 0.5) / BARS * 100 <= pct ? 'bg-brand' : 'bg-ink-muted/35')} style={{ height: `${Math.round(v * 100)}%` }} />
                    ))}
                  </div>
                  {act && (
                    <div className="pointer-events-none absolute inset-y-0 rounded bg-brand/10 ring-1 ring-brand/40"
                      style={{ left: `${(act.start / entry.duration) * 100}%`, width: `${((act.end - act.start) / entry.duration) * 100}%` }} />
                  )}
                  <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-ink" style={{ left: `${pct}%` }} />
                </div>
                {[3, 4, 5].map(lane)}
              </div>
            </div>
            <p className="mt-3 min-h-5 text-xs text-ink-muted">
              {act ? <><span className="font-medium text-ink">{title(act.label)}</span> <span className="font-mono tabular-nums">{fmt(act.start)} – {fmt(act.end)} s</span> · drag the ends, or use ← → (Shift for 100 ms)</>
                : hint ? <><span className="font-medium text-ink">{hint.title ?? hint.key}:</span> {hint.hint}</>
                : 'Pick the sounds below, then drag each one to where it starts and ends.'}
            </p>
          </div>

          <div className="space-y-3 rounded-2xl border border-hairline bg-surface p-4 sm:p-5">
            {GROUPS.map((g) => {
              const opts = menu.filter((o) => o.grp === g && o.key !== 'something else')
              if (!opts.length) return null
              return <Carousel key={g} title={g} options={opts} count={(k) => entry.spans.filter((s) => s.label === k).length} onPick={addSpan} full={entry.spans.length >= 6} />
            })}
            <label className="flex flex-wrap items-center gap-2 border-t border-hairline pt-3 text-sm text-ink-secondary">
              <span>Can’t tell?</span>
              <input
                className="h-9 min-w-0 flex-1 rounded-lg border border-hairline-strong bg-surface px-3 text-sm text-ink placeholder:text-ink-muted focus:border-brand focus:outline-none"
                placeholder="What do you hear?" maxLength={200} value={entry.other}
                onChange={(e) => update((x) => ({ ...x, other: e.target.value }))}
              />
            </label>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={() => go(idx - 1)} disabled={idx === 0 || busy} aria-label="Previous clip">‹ Prev</Button>
            <Button variant="ghost" onClick={() => go(idx + 1)} disabled={idx >= history.length - 1 || busy} aria-label="Next clip">Next ›</Button>
            <span className="flex-1" />
            <Button variant="primary" onClick={submit} disabled={busy || !src}>{entry.submitted ? 'Save changes' : 'Submit'}</Button>
          </div>
        </div>
      )}

      {src && <audio ref={audio} src={src} preload="auto" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />}
      {toast && <div role="status" className="fixed bottom-6 left-1/2 z-30 -translate-x-1/2 rounded-lg bg-ink px-3.5 py-2 text-sm font-medium text-ink-inverse shadow-lg">{toast}</div>}
      {intro && <HeadphonesPrompt onReady={() => { setStarted(true); audio.current?.play().catch(() => {}) }} onClosed={() => setIntro(false)} />}
    </>
  )
}

/** One category's labels in a row that scrolls sideways, with arrows. */
function Carousel({ title, options, count, onPick, full }: { title: string; options: Option[]; count: (k: string) => number; onPick: (o: Option) => void; full: boolean }) {
  const row = useRef<HTMLDivElement>(null)
  const by = (dir: number) => row.current?.scrollBy({ left: dir * row.current.clientWidth * 0.8, behavior: 'smooth' })
  return (
    <div>
      <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{title}</h3>
      <div className="flex items-center gap-1.5">
        <button type="button" aria-label={`Earlier ${title} labels`} onClick={() => by(-1)} className="h-8 w-7 shrink-0 rounded-lg text-ink-muted hover:bg-surface-hover hover:text-ink">‹</button>
        <div ref={row} className="no-scrollbar flex min-w-0 flex-1 snap-x gap-1.5 overflow-x-auto scroll-smooth">
          {options.map((o) => {
            const n = count(o.key)
            return (
              <button key={o.key} type="button" title={o.hint} disabled={full && n === 0} onClick={() => onPick(o)}
                className={cx('inline-flex h-8 shrink-0 snap-start items-center gap-1.5 rounded-lg border px-2.5 text-[13px] transition-colors disabled:opacity-40',
                  n ? 'border-brand-border bg-brand-soft font-medium text-brand' : 'border-hairline bg-surface text-ink hover:bg-surface-hover')}>
                {o.title ?? o.key}
                {n > 1 && <span className="font-mono text-[11px]">×{n}</span>}
              </button>
            )
          })}
        </div>
        <button type="button" aria-label={`More ${title} labels`} onClick={() => by(1)} className="h-8 w-7 shrink-0 rounded-lg text-ink-muted hover:bg-surface-hover hover:text-ink">›</button>
      </div>
    </div>
  )
}
