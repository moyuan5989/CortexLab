import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Trash2,
  Plus,
  Clock,
  Play,
  CheckCircle,
  XCircle,
  ArrowUp,
} from 'lucide-react'
import { useRuns, useDeleteRun } from '../hooks/useRuns'
import { useQueue, useQueueStats, useCancelJob, usePromoteJob } from '../hooks/useQueue'
import StatusBadge from '../components/shared/StatusBadge'
import StatCard from '../components/shared/StatCard'
import { formatLoss, truncate } from '../lib/utils'
import type { Run } from '../api/types'
import type { QueueJob } from '../api/types'

interface UnifiedRow {
  kind: 'queue' | 'run'
  sortKey: number
  queueJob?: QueueJob
  run?: Run
}

export default function Experiments() {
  const { data: runs, isLoading: runsLoading } = useRuns()
  const { data: jobs } = useQueue()
  const { data: stats } = useQueueStats()
  const deleteRun = useDeleteRun()
  const cancelJob = useCancelJob()
  const promoteJob = usePromoteJob()
  const [confirmId, setConfirmId] = useState<string | null>(null)

  function handleDelete(id: string) {
    if (confirmId === id) {
      deleteRun.mutate(id)
      setConfirmId(null)
    } else {
      setConfirmId(id)
    }
  }

  // Merge queue jobs + filesystem runs into a unified list
  const rows = useMemo(() => {
    const result: UnifiedRow[] = []
    const runIdsFromJobs = new Set<string>()

    // Add queue jobs (queued/running first)
    if (jobs) {
      for (const job of jobs) {
        if (job.run_id) runIdsFromJobs.add(job.run_id)

        if (job.status === 'queued' || job.status === 'running') {
          result.push({
            kind: 'queue',
            // queued jobs sort before everything; running next
            sortKey: job.status === 'queued' ? 2_000_000 + (1000 - job.position) : 1_000_000,
            queueJob: job,
          })
        }
      }
    }

    // Add filesystem runs (skip those already represented by a running job)
    if (runs) {
      for (const run of runs) {
        if (runIdsFromJobs.has(run.id) && run.status === 'running') continue
        result.push({
          kind: 'run',
          sortKey: 0, // will be sorted by position in original list
          run,
        })
      }
    }

    // Sort: queued first (highest sortKey), then running, then filesystem runs as-is
    result.sort((a, b) => b.sortKey - a.sortKey)
    return result
  }, [runs, jobs])

  if (runsLoading) {
    return <p className="text-caption text-sm">Loading experiments...</p>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-heading">Experiments</h2>
        <Link
          to="/new"
          className="flex items-center gap-1 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
        >
          <Plus className="h-4 w-4" /> New Training
        </Link>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <StatCard icon={Clock} label="Queued" value={stats.queued} />
          <StatCard icon={Play} label="Running" value={stats.running} />
          <StatCard icon={CheckCircle} label="Completed" value={stats.completed} />
          <StatCard icon={XCircle} label="Failed" value={stats.failed} />
        </div>
      )}

      {rows.length === 0 ? (
        <p className="text-sm text-caption">No training runs found. Start a new training to get started.</p>
      ) : (
        <div className="rounded-lg border border-subtle overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle text-caption">
                <th className="text-left px-4 py-2 font-medium">Run ID</th>
                <th className="text-left px-4 py-2 font-medium">Model</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-right px-4 py-2 font-medium">Progress</th>
                <th className="text-right px-4 py-2 font-medium">Train Loss</th>
                <th className="text-right px-4 py-2 font-medium">Val Loss</th>
                <th className="text-right px-4 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-subtle">
              {rows.map((row) =>
                row.kind === 'queue' ? (
                  <QueueRow
                    key={`q-${row.queueJob!.id}`}
                    job={row.queueJob!}
                    onCancel={() => cancelJob.mutate(row.queueJob!.id)}
                    onPromote={() => promoteJob.mutate(row.queueJob!.id)}
                  />
                ) : (
                  <RunRow
                    key={`r-${row.run!.id}`}
                    run={row.run!}
                    confirmId={confirmId}
                    onDelete={() => handleDelete(row.run!.id)}
                  />
                )
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function QueueRow({ job, onCancel, onPromote }: {
  job: QueueJob
  onCancel: () => void
  onPromote: () => void
}) {
  const modelPath = (job.config?.model as Record<string, unknown>)?.path as string | undefined
  const isStarting = job.status === 'running' && job.run_id && job.run_id.includes('/')

  const statusLabel = job.status === 'queued'
    ? 'queued'
    : isStarting
      ? 'running'
      : job.status

  const runLink = job.run_id && !job.run_id.includes('/')
    ? (
      <Link to={`/experiments/${job.run_id}`} className="text-indigo-400 hover:text-indigo-300">
        {truncate(job.run_id, 24)}
      </Link>
    )
    : (
      <span className="text-caption">
        {job.status === 'queued' ? `Queue #${job.position + 1}` : 'Starting...'}
      </span>
    )

  return (
    <tr className="hover:bg-surface-hover">
      <td className="px-4 py-2">{runLink}</td>
      <td className="px-4 py-2 text-body font-mono text-xs">
        {modelPath ? truncate(modelPath, 30) : '-'}
      </td>
      <td className="px-4 py-2">
        <StatusBadge status={statusLabel} />
      </td>
      <td className="px-4 py-2 text-right text-caption">-</td>
      <td className="px-4 py-2 text-right text-caption">-</td>
      <td className="px-4 py-2 text-right text-caption">-</td>
      <td className="px-4 py-2 text-right">
        <div className="flex items-center justify-end gap-1">
          {job.status === 'queued' && job.position > 0 && (
            <button
              onClick={onPromote}
              className="p-1 text-caption hover:text-label transition-colors"
              title="Move to front"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
          {(job.status === 'queued' || job.status === 'running') && (
            <button
              onClick={onCancel}
              className="p-1 text-caption hover:text-red-400 transition-colors"
              title="Cancel"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
        </div>
      </td>
    </tr>
  )
}

function RunRow({ run, confirmId, onDelete }: {
  run: Run
  confirmId: string | null
  onDelete: () => void
}) {
  return (
    <tr className="hover:bg-surface-hover">
      <td className="px-4 py-2">
        <Link to={`/experiments/${run.id}`} className="text-indigo-400 hover:text-indigo-300">
          {truncate(run.id, 24)}
        </Link>
      </td>
      <td className="px-4 py-2 text-body font-mono text-xs">
        {run.model ? truncate(run.model, 30) : '-'}
      </td>
      <td className="px-4 py-2">
        <StatusBadge status={run.status} />
      </td>
      <td className="px-4 py-2 text-right text-body">
        {run.current_step}/{run.num_iters}
      </td>
      <td className="px-4 py-2 text-right font-mono text-label">
        {formatLoss(run.latest_train_loss)}
      </td>
      <td className="px-4 py-2 text-right font-mono text-label">
        {formatLoss(run.latest_val_loss)}
      </td>
      <td className="px-4 py-2 text-right">
        <button
          className="text-caption hover:text-red-400 transition-colors p-1"
          onClick={onDelete}
          title={confirmId === run.id ? 'Click again to confirm' : 'Delete run'}
        >
          <Trash2 className={`h-4 w-4 ${confirmId === run.id ? 'text-red-400' : ''}`} />
        </button>
      </td>
    </tr>
  )
}
