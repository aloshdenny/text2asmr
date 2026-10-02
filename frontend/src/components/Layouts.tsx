import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import TopBar, { Brand } from './TopBar'
import { ThemeToggle, cx } from './ui'

function Footer() {
  return (
    <footer className="border-t border-hairline">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-6 text-xs text-ink-muted sm:px-6">
        <span>ASMR Board · part of text2asmr</span>
        <Link to="/leaderboard" className="transition-colors hover:text-ink">Leaderboard</Link>
        <Link to="/guide" className="transition-colors hover:text-ink">Sound guide</Link>
        <a href="https://github.com/aloshdenny/text2asmr" className="transition-colors hover:text-ink">GitHub</a>
      </div>
    </footer>
  )
}

/** Top bar, a centred column, footer. */
export function AppLayout({ children, width = 'default' }: { children: ReactNode; width?: 'default' | 'narrow' }) {
  return (
    <div className="flex min-h-screen flex-col bg-canvas">
      <TopBar />
      <main className={cx('mx-auto w-full flex-1 px-4 py-8 sm:px-6 sm:py-10', width === 'narrow' ? 'max-w-2xl' : 'max-w-5xl')}>{children}</main>
      <Footer />
    </div>
  )
}

/** Sign in / sign up / onboarding: full-bleed, brand above a single card, theme toggle in the corner. */
export function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-screen items-center justify-center bg-canvas px-4 py-12 sm:px-6 sm:py-16">
      <ThemeToggle className="absolute right-4 top-4 sm:right-6 sm:top-6" />
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <Brand />
        </div>
        {children}
      </div>
    </div>
  )
}
