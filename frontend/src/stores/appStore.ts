import { create } from 'zustand'

interface AppStore {
  // State
  isLoading: boolean
  error: string | null

  // Actions
  setLoading: (isLoading: boolean) => void
  setError: (error: string | null) => void
}

export const useAppStore = create<AppStore>((set) => ({
  isLoading: false,
  error: null,

  setLoading: (isLoading: boolean) => set({ isLoading }),
  setError: (error: string | null) => set({ error }),
}))
