import type { ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { GuideList } from './components/GuidePanel'
import Nav from './components/Nav'
import { AuthProvider, useAuth } from './lib/auth'
import Home from './pages/Home'
import Label from './pages/Label'
import Leaderboard from './pages/Leaderboard'
import Login from './pages/Login'
import AuthConfirm from './pages/AuthConfirm'
import Profile from './pages/Profile'
import SetPassword from './pages/SetPassword'
import Settings from './pages/Settings'
import Welcome from './pages/Welcome'

function RequireProfile({ children }: { children: ReactNode }) {
  const { session, profile, loading } = useAuth()
  if (loading) return <main className="narrow"><p className="muted">Loading…</p></main>
  if (!session) return <Navigate to="/login" replace />
  if (!profile) return <Navigate to="/welcome" replace />
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Nav />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/label" element={<RequireProfile><Label /></RequireProfile>} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/u/:username" element={<Profile />} />
          <Route
            path="/guide"
            element={
              <main className="narrow">
                <h1>Sound guide</h1>
                <p className="muted">What each label covers, with example clips. Listeners who share one meaning per label make the labels worth more.</p>
                <div className="card"><GuideList /></div>
              </main>
            }
          />
          <Route path="/login" element={<Login />} />
          <Route path="/auth/confirm" element={<AuthConfirm />} />
          <Route path="/set-password" element={<SetPassword />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/welcome" element={<Welcome />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <footer className="footer">
          <span>ASMR Board · part of <a href="https://github.com/aloshdenny/text2asmr">text2asmr</a></span>
        </footer>
      </BrowserRouter>
    </AuthProvider>
  )
}
