import type { LucideIcon } from 'lucide-react'

interface Props {
  icon: LucideIcon
  label: string
  value: string | number
}

export default function StatCard({ icon: Icon, label, value }: Props) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-md bg-zinc-700/50 p-2">
          <Icon className="h-4 w-4 text-zinc-400" />
        </div>
        <div>
          <p className="text-2xl font-semibold text-zinc-50">{value}</p>
          <p className="text-xs text-zinc-500">{label}</p>
        </div>
      </div>
    </div>
  )
}
