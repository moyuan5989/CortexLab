import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: api.getModels })
}

export function useSupportedArchitectures() {
  return useQuery({ queryKey: ['models', 'supported'], queryFn: api.getSupportedArchitectures })
}
