import { Link } from "@tanstack/react-router";
import { Shirt, Sparkles, ImagePlus, Bookmark, Compass } from "lucide-react";
import type { ReactNode } from "react";

const NAV = [
  { to: "/", label: "Wardrobe", icon: Shirt },
  { to: "/discover", label: "Discover", icon: Compass },
  { to: "/generate", label: "Generate", icon: Sparkles },
  { to: "/reference", label: "Reference", icon: ImagePlus },
  { to: "/saved", label: "Saved", icon: Bookmark },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen md:flex">
      {/* Desktop sidebar */}
      <aside className="glass sticky top-0 hidden h-screen w-64 shrink-0 flex-col gap-8 rounded-none border-y-0 border-l-0 p-6 md:flex">
        <Link to="/" className="block">
          <span className="font-display text-3xl tracking-tight">Styla</span>
          <p className="mt-1 text-xs uppercase tracking-[0.2em] text-muted-foreground">
            AI stylist
          </p>
        </Link>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact: to === "/" }}
              className="group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-accent/60 data-[status=active]:bg-accent-soft data-[status=active]:font-medium data-[status=active]:text-primary"
            >
              <Icon className="size-4" />
              {label}
            </Link>
          ))}
        </nav>
        <p className="mt-auto text-xs leading-relaxed text-muted-foreground">
          Prototype preview — outfit analysis is simulated.
        </p>
      </aside>

      <main className="min-w-0 flex-1 pb-28 md:pb-10">
        <header className="glass sticky top-0 z-30 flex items-center justify-between rounded-none border-x-0 border-t-0 px-5 py-3 md:hidden">
          <span className="font-display text-2xl">Styla</span>
          <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            AI stylist
          </span>
        </header>
        <div className="mx-auto w-full max-w-5xl px-5 py-6 md:px-10 md:py-12">{children}</div>
      </main>

      {/* Mobile bottom nav */}
      <nav className="glass fixed inset-x-3 bottom-3 z-40 grid grid-cols-4 gap-1 rounded-3xl p-1.5 md:hidden">
        {NAV.map(({ to, label, icon: Icon }) => (
          <Link
            key={to}
            to={to}
            activeOptions={{ exact: to === "/" }}
            className="flex flex-col items-center gap-1 rounded-2xl px-1 py-2 text-[11px] text-muted-foreground transition-colors data-[status=active]:bg-accent-soft data-[status=active]:text-primary"
          >
            <Icon className="size-[18px]" />
            <span className="truncate">{label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}
