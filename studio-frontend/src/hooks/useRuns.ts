import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'

export function useRuns() {
  return useQuery({ queryKey: ['runs'], queryFn: api.getRuns, refetchInterval: 10_000 })
}

export function useRun(id: string) {
  return useQuery({ queryKey: ['runs', id], queryFn: () => api.getRun(id), enabled: !!id })
}

export function useRunMetrics(id: string) {
  return useQuery({ queryKey: ['runs', id, 'metrics'], queryFn: () => api.getRunMetrics(id), enabled: !!id })
}

export function useRunConfig(id: string) {
  return useQuery({ queryKey: ['runs', id, 'config'], queryFn: () => api.getRunConfig(id), enabled: !!id })
}

export function useRunCheckpoints(id: string) {
  return useQuery({ queryKey: ['runs', id, 'checkpoints'], queryFn: () => api.getRunCheckpoints(id), enabled: !!id })
}

export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.deleteRun(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runs'] }),
  })
}
