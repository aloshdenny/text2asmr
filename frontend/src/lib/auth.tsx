import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './supabase'

export type Profile = { id: string; username: string; display_name: string | null; avatar_url: string | null; onboarded: boolean }

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
  const refreshProfile = useCallback(async () => {
    if (!uid) return
    const { data } = await supabase.from('profiles').select('id, username, display_name, avatar_url, onboarded').eq('id', uid).maybeSingle()
    setProfile(data as Profile | null)
    setProfileOf(uid)
  }, [uid])

  useEffect(() => {
    refreshProfile()
  }, [refreshProfile])

  const loading = !ready || (uid !== null && profileOf !== uid)
  return (
    <AuthContext.Provider value={{ session, profile: uid && profileOf === uid ? profile : null, loading, refreshProfile }}>
      {children}
    </AuthContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => useContext(AuthContext)
