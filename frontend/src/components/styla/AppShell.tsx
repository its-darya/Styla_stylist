import { Link, useLocation, useNavigate } from "@tanstack/react-router";
import { Shirt, Sparkles, ImagePlus, Bookmark, Compass, LogOut } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/styla/auth";

const NAV = [
  { to: "/", label: "Wardrobe", icon: Shirt },
  { to: "/discover", label: "Discover", icon: Compass },
  { to: "/generate", label: "Generate", icon: Sparkles },
  { to: "/reference", label: "Reference", icon: ImagePlus },
  { to: "/saved", label: "Saved", icon: Bookmark },
] as const;

const AUTH_ROUTES = ["/login", "/signup"];

function Loader() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="text-center">
        <span className="font-display text-3xl tracking-tight">Styla</span>
        <p className="mt-2 text-xs uppercase tracking-[0.25em] text-muted-foreground">Loading…</p>
      </div>
    </div>
  );
}

function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-5 py-10">
      <Link to="/login" className="mb-8 text-center">
        <span className="font-display text-4xl tracking-tight">Styla</span>
        <p className="mt-1 text-xs uppercase tracking-[0.25em] text-muted-foreground">AI stylist</p>
      </Link>
      <div className="w-full max-w-md">{children}</div>
    </div>
  );
}

function initials(name: string, email: string) {
  const source = name.trim() || email;
  return source
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

export function AppShell({ children }: { children: ReactNode }) {
  const { status, user, signOut } = useAuth();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const isAuthRoute = AUTH_ROUTES.includes(pathname);

  // Gate: signed-out users go to /login, signed-in users skip the auth pages.
  useEffect(() => {
    if (status === "loading") return;
    if (status === "signed-out" && !isAuthRoute) {
      void navigate({ to: "/login" });
    } else if (status === "signed-in" && isAuthRoute) {
      void navigate({ to: "/" });
    }
  }, [status, isAuthRoute, navigate]);

  if (isAuthRoute) {
    if (status === "signed-in") return <Loader />;
    return <AuthLayout>{children}</AuthLayout>;
  }
  if (status !== "signed-in" || !user) return <Loader />;

  const userChip = (
    <div className="flex items-center gap-3">
      <span className="grid size-9 shrink-0 place-items-center rounded-full bg-accent-soft text-xs font-semibold text-primary">
        {initials(user.name, user.email)}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{user.name || user.email}</p>
        <p className="truncate text-[11px] text-muted-foreground">{user.email}</p>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="rounded-full"
        onClick={signOut}
        aria-label="Sign out"
        title="Sign out"
      >
        <LogOut className="size-4" />
      </Button>
    </div>
  );

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
        <div className="mt-auto">{userChip}</div>
      </aside>

      <main className="min-w-0 flex-1 pb-28 md:pb-10">
        <header className="glass sticky top-0 z-30 flex items-center justify-between rounded-none border-x-0 border-t-0 px-5 py-3 md:hidden">
          <span className="font-display text-2xl">Styla</span>
          <div className="flex items-center gap-2">
            <span className="max-w-[9rem] truncate text-xs text-muted-foreground">
              {user.name || user.email}
            </span>
            <Button variant="ghost" size="icon" className="rounded-full" onClick={signOut} aria-label="Sign out">
              <LogOut className="size-4" />
            </Button>
          </div>
        </header>
        <div className="mx-auto w-full max-w-5xl px-5 py-6 md:px-10 md:py-12">{children}</div>
      </main>

      {/* Mobile bottom nav */}
      <nav className="glass fixed inset-x-3 bottom-3 z-40 grid grid-cols-5 gap-1 rounded-3xl p-1.5 md:hidden">
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
