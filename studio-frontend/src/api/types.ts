// API response types matching the LMForge Studio backend

export interface Run {
  id: string
  path: string
  status: 'completed' | 'running' | 'stopped' | 'unknown'
  model: string | null
  current_step: number
  num_iters: number
  latest_train_loss: number | null
  latest_val_loss: number | null
}

export interface RunDetail extends Run {
  config?: Record<string, unknown>
  manifest?: Record<string, unknown>
  environment?: Record<string, unknown>
}

export interface TrainMetric {
  event: 'train'
  step: number
  train_loss: number
  learning_rate: number
  tokens_per_second: number
  trained_tokens: number
  peak_memory_gb: number
  timestamp: string
}

export interface EvalMetric {
  event: 'eval'
  step: number
  val_loss: number
  timestamp: string
}

export interface Metrics {
  train: TrainMetric[]
  eval: EvalMetric[]
}

export interface CheckpointState {
  schema_version: number
  step: number
  epoch: number
  trained_tokens: number
  best_val_loss: number
  learning_rate: number
  rng_seed: number
}

export interface Checkpoint {
  name: string
  path: string
  is_best: boolean
  state?: CheckpointState
}

export interface Model {
  id: string
  model_id: string
  path: string
  architecture: string
  config: Record<string, unknown>
}

export interface Dataset {
  fingerprint: string
  path: string
  num_samples: number
  total_tokens: number
  min_length: number
  mean_length: number
  max_length: number
  format: string
  model?: string
  template_hash?: string
  tokenizer_hash?: string
  data_hash?: string
}

export interface ActiveTraining {
  track_id: string
  run_id: string
  pid: number
  started_at: string
  config?: Record<string, unknown>
}

export interface GenerationResult {
  text: string
  num_tokens: number
  tokens_per_second: number
  finish_reason: 'stop' | 'length'
}

// WebSocket message types
export interface WsMetricMessage {
  type: 'metric'
  data: TrainMetric | EvalMetric
}

export interface WsTokenMessage {
  type: 'token'
  text: string
}

export interface WsDoneMessage {
  type: 'done'
  stats: { num_tokens: number }
}

export interface WsErrorMessage {
  type: 'error'
  detail: string
}

export interface WsStoppedMessage {
  type: 'stopped'
}

export type WsTrainingMessage = WsMetricMessage | WsErrorMessage | WsStoppedMessage
export type WsInferenceMessage = WsTokenMessage | WsDoneMessage | WsErrorMessage
