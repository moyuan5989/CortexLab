import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FlaskConical,
  Box,
  Database,
  MessageSquare,
  Settings,
} from 'lucide-react'
import { cn } from '../../lib/utils'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/models', label: 'Models', icon: Box },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/playground', label: 'Playground', icon: MessageSquare },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-zinc-950 border-r border-zinc-800 flex flex-col">
      <div className="px-5 py-5">
        <h1 className="text-lg font-bold tracking-tight text-zinc-50">
          LMForge <span className="text-indigo-400 font-normal text-sm">Studio</span>
        </h1>
      </div>

      <nav className="flex-1 px-3 space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-indigo-500/10 text-indigo-400'
                  : 'text-zinc-400 hover:text-zinc-50 hover:bg-zinc-800'
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-5 py-4 text-xs text-zinc-600">
        v0.1.0
      </div>
    </aside>
  )
}
