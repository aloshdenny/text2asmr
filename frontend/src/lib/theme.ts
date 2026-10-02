export type Theme = 'light' | 'dark'

const KEY = 'asmrboard-theme'

/** Dark by default; the toggle switches to light and remembers it. The <html data-theme> attribute (stamped by the
 *  script in index.html before first paint) is the single source of truth -- CSS keyed on it does the rest. */
export function currentTheme(): Theme {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
}

export function toggleTheme() {
  const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark'
  try {
    localStorage.setItem(KEY, next)
  } catch {
    /* private mode: the choice just isn't remembered */
  }
  document.documentElement.dataset.theme = next
  document.documentElement.style.colorScheme = next
}
