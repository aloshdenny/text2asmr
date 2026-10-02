import { Link, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { supabase } from '../lib/supabase'
import Avatar from './Avatar'

export default function Nav() {
  const { session, profile } = useAuth()
  const navigate = useNavigate()
  return (
    <header className="nav">
      <Link to="/" className="brand">
        <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
          <path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 11v2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
        </svg>
        Ear Check
      </Link>
      <nav className="nav-links">
        <NavLink to="/label">Label</NavLink>
        <NavLink to="/leaderboard">Leaderboard</NavLink>
        <NavLink to="/guide">Sound guide</NavLink>
      </nav>
      <div className="nav-me">
        {session && profile ? (
          <>
            <Link to={`/u/${profile.username}`} className="me-link" title="Your contributions">
              <Avatar name={profile.username} url={profile.avatar_url} />
              <span className="me-name">{profile.username}</span>
            </Link>
            <button
              className="btn ghost small"
              onClick={async () => {
                await supabase.auth.signOut()
                navigate('/')
              }}
            >
              Sign out
            </button>
          </>
        ) : session ? (
          <Link to="/welcome" className="btn small">Finish sign-up</Link>
        ) : (
          <Link to="/login" className="btn small">Sign in</Link>
        )}
      </div>
    </header>
  )
}
