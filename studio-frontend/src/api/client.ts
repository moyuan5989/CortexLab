const BASE_URL = '/api/v1'

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  // Runs
  getRuns: () => request<import('./types').Run[]>('/runs'),
  getRun: (id: string) => request<import('./types').RunDetail>(`/runs/${id}`),
  getRunMetrics: (id: string) => request<import('./types').Metrics>(`/runs/${id}/metrics`),
  getRunConfig: (id: string) => request<Record<string, unknown>>(`/runs/${id}/config`),
  getRunCheckpoints: (id: string) => request<import('./types').Checkpoint[]>(`/runs/${id}/checkpoints`),
  deleteRun: (id: string) => request<{ status: string }>(`/runs/${id}`, { method: 'DELETE' }),

  // Models
  getModels: () => request<import('./types').Model[]>('/models'),
  getSupportedArchitectures: () => request<string[]>('/models/supported'),

  // Datasets
  getDatasets: () => request<import('./types').Dataset[]>('/datasets'),
  getDataset: (fp: string) => request<import('./types').Dataset>(`/datasets/${fp}`),
  deleteDataset: (fp: string) => request<{ status: string }>(`/datasets/${fp}`, { method: 'DELETE' }),

  // Training
  startTraining: (config: Record<string, unknown>) =>
    request<{ run_id: string; status: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  stopTraining: (trackId: string) =>
    request<{ status: string }>(`/training/${trackId}/stop`, { method: 'POST' }),
  getActiveTraining: () => request<import('./types').ActiveTraining[]>('/training/active'),

  // Inference
  generate: (body: Record<string, unknown>) =>
    request<import('./types').GenerationResult>('/inference/generate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getInferenceStatus: () => request<{ loaded_model: unknown }>('/inference/status'),
}

export { ApiError }
