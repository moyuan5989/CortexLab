import { Check, X } from 'lucide-react'
import { useModels, useSupportedArchitectures } from '../hooks/useModels'

export default function Models() {
  const { data: models, isLoading } = useModels()
  const { data: supported } = useSupportedArchitectures()

  if (isLoading) return <p className="text-zinc-500 text-sm">Loading models...</p>

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-zinc-50">Models</h2>
        {supported && (
          <p className="text-xs text-zinc-500">
            Supported: {supported.join(', ')}
          </p>
        )}
      </div>

      {!models || models.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No models found in HuggingFace cache.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => {
            const isSupported = supported?.includes(model.architecture)
            const vocabSize = model.config?.vocab_size as number | undefined
            const numLayers = model.config?.num_hidden_layers as number | undefined
            const hiddenSize = model.config?.hidden_size as number | undefined

            return (
              <div
                key={model.id}
                className={`rounded-lg border p-4 ${
                  isSupported
                    ? 'border-zinc-800 bg-zinc-800/50'
                    : 'border-zinc-800/50 bg-zinc-900/50 opacity-60'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-sm font-medium text-zinc-200 break-all">
                    {model.model_id}
                  </h3>
                  {isSupported ? (
                    <Check className="h-4 w-4 text-emerald-400 flex-shrink-0 ml-2" />
                  ) : (
                    <X className="h-4 w-4 text-red-400 flex-shrink-0 ml-2" />
                  )}
                </div>

                <span className="inline-block rounded-full bg-zinc-700/50 px-2 py-0.5 text-xs text-zinc-400 mb-2">
                  {model.architecture}
                </span>

                <div className="text-xs text-zinc-500 space-y-0.5">
                  {numLayers && <p>Layers: {numLayers}</p>}
                  {hiddenSize && <p>Hidden: {hiddenSize}</p>}
                  {vocabSize && <p>Vocab: {vocabSize.toLocaleString()}</p>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
