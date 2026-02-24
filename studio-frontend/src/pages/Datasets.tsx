import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { useDatasets, useDeleteDataset } from '../hooks/useDatasets'
import { formatNumber, truncate } from '../lib/utils'

export default function Datasets() {
  const { data: datasets, isLoading } = useDatasets()
  const deleteDataset = useDeleteDataset()
  const [confirmFp, setConfirmFp] = useState<string | null>(null)

  function handleDelete(fp: string) {
    if (confirmFp === fp) {
      deleteDataset.mutate(fp)
      setConfirmFp(null)
    } else {
      setConfirmFp(fp)
    }
  }

  if (isLoading) return <p className="text-zinc-500 text-sm">Loading datasets...</p>

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-zinc-50">Datasets</h2>

      {!datasets || datasets.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No cached datasets. Run <code className="bg-zinc-800 px-1 rounded text-xs">lmforge prepare</code> to cache a dataset.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((ds) => (
            <div
              key={ds.fingerprint}
              className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="text-sm font-mono text-zinc-300">
                  {truncate(ds.fingerprint, 16)}
                </h3>
                <button
                  className="text-zinc-500 hover:text-red-400 transition-colors p-1"
                  onClick={() => handleDelete(ds.fingerprint)}
                  title={confirmFp === ds.fingerprint ? 'Click again to confirm' : 'Delete'}
                >
                  <Trash2
                    className={`h-4 w-4 ${confirmFp === ds.fingerprint ? 'text-red-400' : ''}`}
                  />
                </button>
              </div>

              <span className="inline-block rounded-full bg-zinc-700/50 px-2 py-0.5 text-xs text-zinc-400 mb-3">
                {ds.format}
              </span>

              <div className="text-xs text-zinc-500 space-y-1">
                <p>
                  Samples: <span className="text-zinc-300">{ds.num_samples.toLocaleString()}</span>
                </p>
                <p>
                  Tokens: <span className="text-zinc-300">{formatNumber(ds.total_tokens, 0)}</span>
                </p>
                <p>
                  Length: <span className="text-zinc-300">
                    {ds.min_length} / {Math.round(ds.mean_length)} / {ds.max_length}
                  </span>
                  <span className="text-zinc-600 ml-1">(min/avg/max)</span>
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
