import { useEffect, useState } from 'react'
import { Mascot } from './TopBar'
import { Button, cx } from './ui'

/** The headphones prompt shows once per visit, whichever labeller is open (its click is also what lets audio play). */
export const headphonesSeen = () => { try { return sessionStorage.getItem('asmrboard-headphones') === '1' } catch { return false } }
export const markHeadphonesSeen = () => { try { sessionStorage.setItem('asmrboard-headphones', '1') } catch { /* private mode */ } }

/** Shown when the labeller opens: asks for headphones, and its button is the click that lets the first clip play.
 *  onReady fires at once (start playback inside the click); the dialog then animates out and calls onClosed. */
export default function HeadphonesPrompt({ onReady, onClosed }: { onReady: () => void; onClosed: () => void }) {
  const [leaving, setLeaving] = useState(false)
  const ready = () => {
    if (leaving) return
    onReady()
    markHeadphonesSeen()
    setLeaving(true)
    window.setTimeout(onClosed, 220)
  }
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === 'Escape' && ready()
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  })
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="hp-title"
      className={cx('hp-backdrop fixed inset-0 z-40 flex items-center justify-center bg-black/45 px-4 backdrop-blur-[3px]', leaving && 'hp-out')}
    >
      <div className="hp-panel w-full max-w-sm rounded-3xl border-2 border-hairline bg-surface-raised px-6 pb-6 pt-7 text-center">
        <div className="flex justify-center"><span className="critter-bob overflow-hidden rounded-3xl"><Mascot size={96} /></span></div>
        <h2 id="hp-title" className="mt-4 text-2xl font-black tracking-tight text-ink">Put your headphones on</h2>
        <p className="mt-2 text-[15px] font-semibold leading-relaxed text-ink-secondary">
          These sounds are soft and recorded close to the mic. Headphones or earphones bring out the small details that
          tell one sound from another.
        </p>
        <Button variant="primary" size="lg" className="mt-6 w-full" onClick={ready} autoFocus>
          I’m ready
        </Button>
      </div>
    </div>
  )
}
