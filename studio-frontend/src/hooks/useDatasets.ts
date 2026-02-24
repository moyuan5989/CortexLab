import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: api.getDatasets })
}

export function useDataset(fingerprint: string) {
  return useQuery({
    queryKey: ['datasets', fingerprint],
    queryFn: () => api.getDataset(fingerprint),
    enabled: !!fingerprint,
  })
}

export function useDeleteDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (fp: string) => api.deleteDataset(fp),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['datasets'] }),
  })
}
