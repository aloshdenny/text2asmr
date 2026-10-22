import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import TopBar, { Brand } from './TopBar'
import { Icon, ThemeToggle, cx } from './ui'

function Footer() {
  return (
    <footer className="border-t-2 border-hairline">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-6 text-xs font-bold text-ink-muted sm:px-6">
        <span>ASMR Board</span>
        <Link to="/leaderboard" className="transition-colors hover:text-ink">Leaderboard</Link>
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

/** Sign in / sign up / onboarding: one narrow column on the bare page. A close button back home in the top left, the
 *  other way in (log in / sign up) in the top right, the theme toggle beside it. */
export function AuthLayout({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="relative flex min-h-screen flex-col items-center bg-canvas px-4 pb-12 pt-20 sm:px-6">
      <Link to="/" aria-label="Back to the home page" className="absolute left-3 top-3 rounded-xl p-2.5 text-ink-muted transition-colors hover:bg-surface-hover hover:text-ink sm:left-5 sm:top-5">
        <Icon className="h-5 w-5"><path d="M18 6 6 18M6 6l12 12" /></Icon>
      </Link>
      <div className="absolute right-3 top-3 flex items-center gap-2 sm:right-5 sm:top-5">
        <ThemeToggle />
        {action}
      </div>
      <div className="w-full max-w-[380px]">
        <div className="mb-6 flex justify-center"><Brand mascotOnly /></div>
        {children}
      </div>
    </div>
  )
}
