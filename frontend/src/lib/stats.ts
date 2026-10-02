import { useEffect, useState } from 'react'
import { supabase, timeZone } from './supabase'

export type UserStats = { username: string; display_name: string | null; avatar_url: string | null; joined: string; labelled: number; rank: number | null }
export type BoardRow = { rank: number; username: string; display_name: string | null; avatar_url: string | null; labelled: number; last_at: string }

/** A listener's public stats and per-day counts. stats: undefined while loading, null if there is no such listener. */
export function useUserStats(username: string | undefined, reloadKey = 0) {
  const [stats, setStats] = useState<UserStats | null | undefined>(undefined)
  const [counts, setCounts] = useState<Map<string, number>>(new Map())
  useEffect(() => {
    if (!username) return
    let live = true
    supabase.rpc('profile_stats', { p_username: username }).then(({ data }) => live && setStats((data as UserStats[] | null)?.[0] ?? null))
    supabase.rpc('contributions', { p_username: username, p_tz: timeZone }).then(({ data }) => {
      if (live) setCounts(new Map(((data as { day: string; labelled: number }[] | null) ?? []).map((r) => [r.day, r.labelled])))
    })
    return () => {
      live = false
    }
  }, [username, reloadKey])
  return { stats, counts }
}

export function useLeaderboard(days: number | null, limit: number) {
  const [rows, setRows] = useState<BoardRow[] | null>(null)
  useEffect(() => {
    let live = true
    supabase.rpc('leaderboard', { p_days: days, p_limit: limit }).then(({ data }) => live && setRows((data as BoardRow[] | null) ?? []))
    return () => {
      live = false
    }
  }, [days, limit])
  return rows
}

export const since = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
