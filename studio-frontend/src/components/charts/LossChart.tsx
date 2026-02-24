import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { TrainMetric, EvalMetric } from '../../api/types'

interface Props {
  trainMetrics: TrainMetric[]
  evalMetrics: EvalMetric[]
}

export default function LossChart({ trainMetrics, evalMetrics }: Props) {
  // Merge into a single array keyed by step
  const byStep = new Map<number, { step: number; train_loss?: number; val_loss?: number }>()

  for (const m of trainMetrics) {
    const entry = byStep.get(m.step) || { step: m.step }
    entry.train_loss = m.train_loss
    byStep.set(m.step, entry)
  }
  for (const m of evalMetrics) {
    const entry = byStep.get(m.step) || { step: m.step }
    entry.val_loss = m.val_loss
    byStep.set(m.step, entry)
  }

  const data = Array.from(byStep.values()).sort((a, b) => a.step - b.step)

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-zinc-500 text-sm">
        No metrics available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis dataKey="step" stroke="#71717a" tick={{ fontSize: 12 }} />
        <YAxis stroke="#71717a" tick={{ fontSize: 12 }} />
        <Tooltip
          contentStyle={{
            backgroundColor: '#18181b',
            border: '1px solid #3f3f46',
            borderRadius: '6px',
            fontSize: '12px',
          }}
        />
        <Legend wrapperStyle={{ fontSize: '12px' }} />
        <Line
          type="monotone"
          dataKey="train_loss"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          name="Train Loss"
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="val_loss"
          stroke="#10b981"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ r: 3, fill: '#10b981' }}
          name="Val Loss"
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
