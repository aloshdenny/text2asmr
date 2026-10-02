import { useEffect, useState } from 'react'
import { guideUrl, loadGuide, loadMenu, type GuideEntry, type Option } from '../lib/supabase'
import { Button, Kbd } from './ui'

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
    <div className="space-y-6">
      {(['Voice', 'Triggers', 'Other'] as const).map((g) => (
        <section key={g}>
          <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{g}</h3>
          <ul className="divide-y divide-hairline">
            {menu
              .filter((o) => o.grp === g)
              .map((o) => (
                <li key={o.key} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink">{o.key}</p>
                    <p className="text-sm text-ink-muted">{o.hint}</p>
                  </div>
                  <div className="flex shrink-0 flex-col gap-1.5">
                    {(examples.get(o.key) ?? []).map((c, i) => (
                      <audio key={c} controls preload="none" src={guideUrl(c)} aria-label={`${o.key} example ${i + 1}`} className="h-8 w-56" />
                    ))}
                  </div>
                </li>
              ))}
          </ul>
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
    <div className="fixed inset-0 z-30 flex justify-end bg-black/30" onClick={onClose}>
      <aside className="scroll-slim h-full w-full max-w-md overflow-y-auto border-l border-hairline bg-surface-raised p-5" onClick={(e) => e.stopPropagation()} aria-label="Sound guide">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-ink">Sound guide</h2>
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close <Kbd>Esc</Kbd>
          </Button>
        </div>
        <GuideList />
      </aside>
    </div>
  )
}
