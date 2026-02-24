interface Props {
  label: string
  value: string | number
  unit?: string
}

export default function MetricCard({ label, value, unit }: Props) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-3">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-lg font-semibold text-zinc-50">
        {value}
        {unit && <span className="text-sm font-normal text-zinc-400 ml-1">{unit}</span>}
      </p>
    </div>
  )
}
