import { useState, useEffect } from 'react'

export default function Settings() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('lmforge-theme') as 'dark' | 'light') || 'dark'
  })

  useEffect(() => {
    localStorage.setItem('lmforge-theme', theme)
    document.documentElement.classList.toggle('light', theme === 'light')
  }, [theme])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-zinc-50">Settings</h2>

      <div className="max-w-lg space-y-6">
        {/* Theme */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
          <label className="block text-sm font-medium text-zinc-300 mb-2">Theme</label>
          <div className="flex gap-2">
            <button
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                theme === 'dark'
                  ? 'bg-indigo-600 text-white'
                  : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
              onClick={() => setTheme('dark')}
            >
              Dark
            </button>
            <button
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                theme === 'light'
                  ? 'bg-indigo-600 text-white'
                  : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
              onClick={() => setTheme('light')}
            >
              Light
            </button>
          </div>
        </div>

        {/* Paths */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
          <label className="block text-sm font-medium text-zinc-300 mb-2">Run Directory</label>
          <p className="text-sm text-zinc-400 font-mono bg-zinc-900 rounded px-3 py-2">
            ~/.lmforge/runs
          </p>
        </div>

        <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
          <label className="block text-sm font-medium text-zinc-300 mb-2">API Base URL</label>
          <p className="text-sm text-zinc-400 font-mono bg-zinc-900 rounded px-3 py-2">
            {window.location.origin}
          </p>
        </div>

        {/* Info */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
          <label className="block text-sm font-medium text-zinc-300 mb-2">About</label>
          <div className="text-sm text-zinc-500 space-y-1">
            <p>LMForge Studio v0.1.0</p>
            <p>LoRA SFT training framework for MLX on Apple Silicon</p>
            <p>
              API docs:{' '}
              <a
                href="/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-400 hover:text-indigo-300"
              >
                /docs
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
