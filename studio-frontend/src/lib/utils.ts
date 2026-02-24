export function formatNumber(n: number, decimals = 1): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(decimals)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(decimals)}K`
  return n.toFixed(decimals)
}

export function formatLoss(loss: number | null | undefined): string {
  if (loss == null) return '-'
  return loss.toFixed(4)
}

export function formatTokPerSec(tps: number | null | undefined): string {
  if (tps == null) return '-'
  return `${tps.toFixed(0)} tok/s`
}

export function formatMemory(gb: number | null | undefined): string {
  if (gb == null) return '-'
  return `${gb.toFixed(1)} GB`
}

export function formatDate(iso: string | undefined): string {
  if (!iso) return '-'
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s
  return s.slice(0, maxLen - 1) + '\u2026'
}

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ')
}

export function getWsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}
