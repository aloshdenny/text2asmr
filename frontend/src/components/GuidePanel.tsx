import { useEffect, useState } from 'react'
import { guideUrl, loadGuide, loadMenu, type GuideEntry, type Option } from '../lib/supabase'

/** Every label with what it covers, and example clips where the guide has them. */
export function GuideList() {
  const [menu, setMenu] = useState<Option[]>([])
  const [guide, setGuide] = useState<GuideEntry[]>([])
  useEffect(() => {
    loadMenu().then(setMenu, () => {})
    loadGuide().then(setGuide)
  }, [])
  const examples = new Map(guide.map((g) => [g.label, g.clips]))
  return (
    <div className="guide-list">
      {(['Voice', 'Triggers', 'Other'] as const).map((g) => (
        <section key={g}>
          <h3>{g}</h3>
          {menu
            .filter((o) => o.grp === g)
            .map((o) => (
              <div key={o.key} className="guide-row">
                <div>
                  <b>{o.key}</b>
                  <p className="muted">{o.hint}</p>
                </div>
                <div className="guide-clips">
                  {(examples.get(o.key) ?? []).map((c, i) => (
                    <audio key={c} controls preload="none" src={guideUrl(c)} aria-label={`${o.key} example ${i + 1}`} />
                  ))}
                </div>
              </div>
            ))}
        </section>
      ))}
    </div>
  )
}

export default function GuidePanel({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const f = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', f)
    return () => window.removeEventListener('keydown', f)
  }, [onClose])
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Sound guide">
        <div className="drawer-head">
          <h2>Sound guide</h2>
          <button className="btn ghost small" onClick={onClose}>Close <kbd>Esc</kbd></button>
        </div>
        <GuideList />
      </aside>
    </div>
  )
}
