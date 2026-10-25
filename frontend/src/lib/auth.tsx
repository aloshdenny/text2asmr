import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import type { Session } from '@supabase/supabase-js'
import { clearReferral, pendingReferral } from './referral'
import { supabase } from './supabase'

/** What to tell someone the database turned away: an account works on two devices (P0004), and a browser that keeps
 *  no device id cannot be counted (P0003). */
export function deviceRefusal(code?: string): string | null {
  if (code === 'P0004') return 'This account is already in use on two other devices. Each account can be used on two devices.'
  if (code === 'P0003') return 'This browser is blocking site data, so it can’t be recognised. Allow site data for asmrboard.vercel.app, then reload.'
  return null
}

export type Profile = { id: string; username: string; display_name: string | null; avatar_url: string | null; onboarded: boolean; guest: boolean }

type AuthState = {
  session: Session | null
  profile: Profile | null
  /** true until we know both who is signed in and whether they have a profile */
  loading: boolean
  refreshProfile: () => Promise<void>
}

const AuthContext = createContext<AuthState>({ session: null, profile: null, loading: true, refreshProfile: async () => {} })

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [ready, setReady] = useState(false)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [profileOf, setProfileOf] = useState<string | null>(null)

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setReady(true)
    })
    // only store the session here: calling Supabase inside this callback can deadlock the auth client
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s))
    return () => data.subscription.unsubscribe()
  }, [])

  const uid = session?.user.id ?? null
  const latest = useRef(0)
  // reads the session itself rather than a captured uid: right after a guest signs in, the caller's render still has
  // the old (empty) session, and a fetch started before start_guest made the profile must not overwrite a newer one
  const refreshProfile = useCallback(async () => {
    const call = ++latest.current
    const { data: auth } = await supabase.auth.getSession()
    const id = auth.session?.user.id
    if (!id) return
    const { data } = await supabase.from('profiles').select('id, username, display_name, avatar_url, onboarded, guest').eq('id', id).maybeSingle()
    if (call !== latest.current) return
    setProfile(data as Profile | null)
    setProfileOf(id)
  }, [])

  useEffect(() => {
    refreshProfile()
  }, [uid, refreshProfile])

  const profileId = profile?.id
  // an account can be used on two devices: a third is signed out as soon as it shows up (the database keeps count)
  useEffect(() => {
    if (!profileId) return
    supabase.rpc('claim_device').then(async ({ error }) => {
      const why = deviceRefusal(error?.code)
      if (!why) return
      try { sessionStorage.setItem('asmrboard:signed-out', why) } catch { /* the login page just won't say why */ }
      await supabase.auth.signOut({ scope: 'local' })
    })
  }, [profileId])

  // a referral link followed before this profile existed is credited once the profile is there
  useEffect(() => {
    const ref = profileId && pendingReferral()
    if (ref) supabase.rpc('set_referral', { p_ref: ref }).then(({ error }) => { if (!error) clearReferral() })
  }, [profileId])

  const loading = !ready || (uid !== null && profileOf !== uid)
  return (
    <AuthContext.Provider value={{ session, profile: uid && profileOf === uid ? profile : null, loading, refreshProfile }}>
      {children}
    </AuthContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => useContext(AuthContext)
