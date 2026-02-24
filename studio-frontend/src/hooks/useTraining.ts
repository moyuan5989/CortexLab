import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'

export function useActiveTraining() {
  return useQuery({ queryKey: ['training', 'active'], queryFn: api.getActiveTraining, refetchInterval: 5_000 })
}

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (config: Record<string, unknown>) => api.startTraining(config),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'active'] })
      qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}

export function useStopTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (trackId: string) => api.stopTraining(trackId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'active'] })
      qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}
