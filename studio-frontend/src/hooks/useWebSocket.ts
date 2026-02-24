import { useEffect, useRef, useCallback, useState } from 'react'
import { getWsUrl } from '../lib/utils'

interface UseWebSocketOptions {
  url: string
  onMessage?: (data: unknown) => void
  enabled?: boolean
  reconnectDelay?: number
}

export function useWebSocket({ url, onMessage, enabled = true, reconnectDelay = 3000 }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const close = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!enabled) return

    let reconnectTimer: ReturnType<typeof setTimeout>
    let stopped = false

    function connect() {
      if (stopped) return
      const ws = new WebSocket(getWsUrl(url))
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!stopped) reconnectTimer = setTimeout(connect, reconnectDelay)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          onMessageRef.current?.(data)
        } catch { /* ignore non-JSON */ }
      }
    }

    connect()

    return () => {
      stopped = true
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
      wsRef.current = null
      setConnected(false)
    }
  }, [url, enabled, reconnectDelay])

  return { connected, send, close }
}
