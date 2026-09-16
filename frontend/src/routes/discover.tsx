import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Eye, Heart, Plus, Undo2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { LOOKS as posts, type Look as DiscoverPost } from "@/lib/styla/looks";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/discover")({
  head: () => ({
    meta: [
      { title: "Discover — Styla" },
      { name: "description", content: "Get inspired by the community's fashion combinations." },
    ],
  }),
  component: DiscoverPage,
});

const ALL = "All";
const LIKES_KEY = "styla.discover.likes";
const HIDDEN_KEY = "styla.discover.hidden";

/** Ids remembered in this browser; a blocked storage box just reads empty. */
function readIds(key: string): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = JSON.parse(window.localStorage.getItem(key) ?? "[]");
    return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

function writeIds(key: string, ids: string[]) {
  try {
    window.localStorage.setItem(key, JSON.stringify(ids));
  } catch {
    // Losing the list only costs the remembered hearts and hidden looks.
  }
}

function DiscoverPage() {
  const navigate = useNavigate();
  const [active, setActive] = useState(ALL);
  const [zoomed, setZoomed] = useState<DiscoverPost | null>(null);
  const [liked, setLiked] = useState<string[]>([]);
  const [hidden, setHidden] = useState<string[]>([]);
  const [showRemoved, setShowRemoved] = useState(false);

  // Read after mount, so the server-rendered markup matches the first paint.
  useEffect(() => {
    setLiked(readIds(LIKES_KEY));
    setHidden(readIds(HIDDEN_KEY));
  }, []);

  const styles = useMemo(() => [ALL, ...Array.from(new Set(posts.map((p) => p.style)))], []);

  const inView = posts.filter((p) =>
    showRemoved ? hidden.includes(p.id) : !hidden.includes(p.id),
  );
  const visible = active === ALL ? inView : inView.filter((p) => p.style === active);

  function saveLiked(ids: string[]) {
    setLiked(ids);
    writeIds(LIKES_KEY, ids);
  }

  function saveHidden(ids: string[]) {
    setHidden(ids);
    writeIds(HIDDEN_KEY, ids);
  }

  /** Hand the look to Reference, which matches it against the wardrobe. */
  function checkInReference(post: DiscoverPost) {
    setZoomed(null);
    toast.success("Checking this look against your wardrobe");
    void navigate({ to: "/reference", search: { image: post.imageUrl } });
  }

  /**
   * Favouriting only saves the look. Checking it against the wardrobe is a
   * separate, deliberate step — saving something you like shouldn't drag you
   * off the page.
   */
  function toggleLike(post: DiscoverPost) {
    if (liked.includes(post.id)) {
      saveLiked(liked.filter((id) => id !== post.id));
      toast("Removed from favourites");
      return;
    }
    saveLiked([...liked, post.id]);
    toast.success("Saved to favourites", {
      action: { label: "Check it", onClick: () => checkInReference(post) },
    });
  }

  function removeLook(post: DiscoverPost) {
    setZoomed(null);
    saveHidden([...hidden, post.id]);
    toast("Removed from Discover", {
      action: { label: "Undo", onClick: () => saveHidden(hidden.filter((id) => id !== post.id)) },
    });
  }

  function restoreLook(post: DiscoverPost) {
    const left = hidden.filter((id) => id !== post.id);
    saveHidden(left);
    if (left.length === 0) setShowRemoved(false);
    toast.success("Added back to Discover");
  }

  return (
    <div className="space-y-8 pb-12">
      <header className="text-center pt-8 pb-2">
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Inspiration</p>
        <h1 className="mt-4 text-4xl md:text-5xl font-display tracking-tight">Discover Styles</h1>
        <p className="mt-4 max-w-xl mx-auto text-muted-foreground text-base">
          Tap a look to see it bigger, save the ones you like, and check any of them against your
          wardrobe when you want to.
        </p>
      </header>

      {/* Style filter pills, the active one filled */}
      <div className="flex flex-wrap justify-center gap-2">
        {styles.map((style) => (
          <button
            key={style}
            type="button"
            onClick={() => setActive(style)}
            className={cn(
              "rounded-full border px-4 py-1.5 text-sm transition",
              style === active
                ? "border-foreground bg-foreground text-background"
                : "border-border text-muted-foreground hover:border-foreground hover:text-foreground",
            )}
          >
            {style}
          </button>
        ))}
      </div>

      {/* Only shown once something has been removed */}
      {hidden.length > 0 && (
        <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
          <span className="text-muted-foreground">
            {hidden.length} look{hidden.length === 1 ? "" : "s"} removed
          </span>
          <Button
            size="sm"
            variant={showRemoved ? "default" : "outline"}
            className="rounded-full"
            onClick={() => setShowRemoved((on) => !on)}
          >
            {showRemoved ? "Back to Discover" : "Show removed"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="rounded-full"
            onClick={() => {
              saveHidden([]);
              setShowRemoved(false);
              toast.success("All looks are back");
            }}
          >
            <Undo2 className="size-3.5" /> Add all back
          </Button>
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
        {visible.map((post) => {
          const isLiked = liked.includes(post.id);
          const isRemoved = hidden.includes(post.id);
          return (
            <div key={post.id} className="group relative">
              <button
                type="button"
                onClick={() => setZoomed(post)}
                aria-label={`See this ${post.style} look bigger`}
                className="block w-full overflow-hidden rounded-lg bg-white aspect-[3/5] transition-all duration-300 hover:shadow-lg hover:-translate-y-0.5"
              >
                <img
                  src={post.imageUrl}
                  alt={`${post.style} look`}
                  loading="lazy"
                  className={cn(
                    "h-full w-full object-contain transition-transform duration-500 group-hover:scale-[1.03]",
                    isRemoved && "opacity-50",
                  )}
                />
              </button>

              {isRemoved ? (
                <button
                  type="button"
                  onClick={() => restoreLook(post)}
                  aria-label="Add this look back to Discover"
                  title="Add back to Discover"
                  className="absolute left-2 top-2 grid size-8 place-items-center rounded-full bg-white/85 shadow-sm backdrop-blur transition hover:bg-white"
                >
                  <Plus className="size-4 text-primary" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => removeLook(post)}
                  aria-label="Remove this look from Discover"
                  title="Remove from Discover"
                  className="absolute left-2 top-2 grid size-8 place-items-center rounded-full bg-white/85 opacity-0 shadow-sm backdrop-blur transition hover:bg-white focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <X className="size-4 text-foreground/60" />
                </button>
              )}

              <button
                type="button"
                onClick={() => toggleLike(post)}
                aria-label={isLiked ? "Remove from favourites" : "Save to favourites"}
                title={isLiked ? "Remove from favourites" : "Save to favourites"}
                className="absolute right-2 top-2 grid size-8 place-items-center rounded-full bg-white/85 shadow-sm backdrop-blur transition hover:bg-white"
              >
                <Heart
                  className={cn("size-4", isLiked ? "fill-primary text-primary" : "text-foreground/60")}
                />
              </button>

              <div className="mt-2 flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium text-foreground">{post.style}</p>
                <span className="flex shrink-0 items-center text-[11px] text-muted-foreground">
                  <Eye className="mr-0.5 size-3" />
                  {post.views}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {visible.length === 0 && (
        <p className="py-10 text-center text-sm text-muted-foreground">
          {showRemoved
            ? "Nothing removed in this style."
            : "Every look in this style is removed. Use “Show removed” to bring one back."}
        </p>
      )}

      {/* Bigger view */}
      <Dialog open={zoomed !== null} onOpenChange={(open) => !open && setZoomed(null)}>
        <DialogContent className="max-h-[92vh] max-w-2xl overflow-y-auto rounded-3xl">
          <DialogHeader>
            <DialogTitle className="font-display text-2xl">{zoomed?.style}</DialogTitle>
          </DialogHeader>
          {zoomed && (
            <>
              <img
                src={zoomed.imageUrl}
                alt={`${zoomed.style} look`}
                className="max-h-[65vh] w-full rounded-2xl bg-white object-contain"
              />
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className="flex items-center text-xs text-muted-foreground">
                  <Eye className="mr-1 size-3.5" />
                  {zoomed.views}
                </span>
                <div className="flex flex-wrap gap-2">
                  {hidden.includes(zoomed.id) ? (
                    <Button
                      variant="outline"
                      className="rounded-full"
                      onClick={() => restoreLook(zoomed)}
                    >
                      <Plus className="size-4" /> Add back
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      className="rounded-full"
                      onClick={() => removeLook(zoomed)}
                    >
                      <X className="size-4" /> Remove
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    className="rounded-full"
                    onClick={() => toggleLike(zoomed)}
                  >
                    <Heart
                      className={cn(
                        "size-4",
                        liked.includes(zoomed.id) ? "fill-primary text-primary" : "",
                      )}
                    />
                    {liked.includes(zoomed.id) ? "Favourited" : "Favourite"}
                  </Button>
                  <Button className="rounded-full" onClick={() => checkInReference(zoomed)}>
                    Check in Reference
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
