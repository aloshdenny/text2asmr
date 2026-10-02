import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'
import Avatar from './Avatar'
import { ButtonLink, Icon, ThemeToggle, cx } from './ui'

export function Brand({ className }: { className?: string }) {
  const { session, profile } = useAuth()
  return (
    <Link to={session && profile?.onboarded ? '/dashboard' : '/'} className={cx('inline-flex items-center gap-2 text-[15px] font-semibold tracking-tight text-ink', className)}>
      <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] text-brand" aria-hidden="true">
        <path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 11v2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" fill="none" />
      </svg>
      ASMR Board
    </Link>
  )
}

function AccountMenu() {
  const { profile } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => setOpen(false), [location.pathname])
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !ref.current?.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', close)
    window.addEventListener('keydown', close)
    return () => {
      window.removeEventListener('mousedown', close)
      window.removeEventListener('keydown', close)
    }
  }, [open])
  if (!profile) return null
  const item = 'flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-[13px] text-ink-secondary transition-colors hover:bg-surface-hover hover:text-ink'
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex items-center gap-2 rounded-lg py-1 pl-1 pr-2 transition-colors hover:bg-surface-hover"
      >
        <Avatar name={profile.username} url={profile.avatar_url} size={26} />
        <span className="hidden text-[13px] font-medium text-ink sm:inline">{profile.username}</span>
        <Icon className="h-3.5 w-3.5 text-ink-muted"><path d="m6 9 6 6 6-6" /></Icon>
      </button>
      {open && (
        <div role="menu" className="absolute right-0 top-full z-20 mt-1.5 w-52 rounded-xl border border-hairline bg-surface-raised p-1 shadow-[0_10px_30px_-12px_rgb(0_0_0/0.25)]">
          <div className="px-2.5 pb-1.5 pt-1 text-xs text-ink-muted">Signed in as <span className="font-medium text-ink">@{profile.username}</span></div>
          <Link role="menuitem" to="/dashboard" className={item}>
            <Icon><path d="M3 13h4v7H3zM10 8h4v12h-4zM17 4h4v16h-4z" /></Icon>Dashboard
          </Link>
          <Link role="menuitem" to="/label" className={item}>
            <Icon><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></Icon>Label clips
          </Link>
          <Link role="menuitem" to="/settings" className={item}>
            <Icon><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></Icon>Edit profile
          </Link>
          <div className="my-1 h-px bg-hairline" />
          <button
            role="menuitem"
            className={item}
            onClick={async () => {
              navigate('/')
              await supabase.auth.signOut()
            }}
          >
            <Icon><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="m16 17 5-5-5-5M21 12H9" /></Icon>Sign out
          </button>
        </div>
      )}
    </div>
  )
}

export default function TopBar() {
  const { session, profile, loading } = useAuth()
  return (
    <header className="sticky top-0 z-10 border-b border-hairline bg-surface/85 backdrop-blur-md">
      <div className="mx-auto flex h-[var(--topbar-height)] max-w-5xl items-center gap-3 px-4 sm:px-6">
        <Brand />
        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
          {loading ? null : session && profile ? (
            <AccountMenu />
          ) : session ? (
            <ButtonLink to="/welcome" size="sm" variant="primary">Finish sign-up</ButtonLink>
          ) : (
            <>
              <ButtonLink to="/login" size="sm" variant="ghost">Log in</ButtonLink>
              <ButtonLink to="/signup" size="sm" variant="primary">Sign up</ButtonLink>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
