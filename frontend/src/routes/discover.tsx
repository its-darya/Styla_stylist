import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Eye, Heart, MapPin } from "lucide-react";
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

function readLikes(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = JSON.parse(window.localStorage.getItem(LIKES_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

function DiscoverPage() {
  const navigate = useNavigate();
  const [active, setActive] = useState(ALL);
  const [zoomed, setZoomed] = useState<DiscoverPost | null>(null);
  const [liked, setLiked] = useState<string[]>([]);

  // Likes are read after mount so the server-rendered markup matches.
  useEffect(() => setLiked(readLikes()), []);

  const styles = useMemo(() => [ALL, ...Array.from(new Set(posts.map((p) => p.style)))], []);
  const visible = active === ALL ? posts : posts.filter((p) => p.style === active);

  function persist(ids: string[]) {
    setLiked(ids);
    try {
      window.localStorage.setItem(LIKES_KEY, JSON.stringify(ids));
    } catch {
      // A blocked storage box only costs the remembered hearts.
    }
  }

  /** Hand the look to Reference, which matches it against the wardrobe. */
  function checkInReference(post: DiscoverPost) {
    setZoomed(null);
    toast.success("Checking this look against your wardrobe");
    void navigate({ to: "/reference", search: { image: post.imageUrl } });
  }

  /** Liking sends the look straight to Reference; unliking just forgets it. */
  function toggleLike(post: DiscoverPost) {
    if (liked.includes(post.id)) {
      persist(liked.filter((id) => id !== post.id));
      return;
    }
    persist([...liked, post.id]);
    checkInReference(post);
  }

  return (
    <div className="space-y-8 pb-12">
      <header className="text-center pt-8 pb-2">
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Inspiration</p>
        <h1 className="mt-4 text-4xl md:text-5xl font-display tracking-tight">Discover Styles</h1>
        <p className="mt-4 max-w-xl mx-auto text-muted-foreground text-base">
          Tap a look to see it bigger, or like it to check it against your wardrobe.
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

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
        {visible.map((post) => {
          const isLiked = liked.includes(post.id);
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
                  className="h-full w-full object-contain transition-transform duration-500 group-hover:scale-[1.03]"
                />
              </button>

              <button
                type="button"
                onClick={() => toggleLike(post)}
                aria-label={isLiked ? "Remove like" : "Like and check in Reference"}
                title={isLiked ? "Remove like" : "Like and check in Reference"}
                className="absolute right-2 top-2 grid size-8 place-items-center rounded-full bg-white/85 shadow-sm backdrop-blur transition hover:bg-white"
              >
                <Heart
                  className={cn("size-4", isLiked ? "fill-primary text-primary" : "text-foreground/60")}
                />
              </button>

              <p className="mt-2 truncate text-sm font-medium text-foreground">{post.style}</p>
              <div className="mt-0.5 flex items-center justify-between gap-1 text-[11px] text-muted-foreground">
                <span className="flex min-w-0 items-center">
                  <MapPin className="mr-0.5 size-3 shrink-0" />
                  <span className="truncate">{post.location}</span>
                </span>
                <span className="flex shrink-0 items-center">
                  <Eye className="mr-0.5 size-3" />
                  {post.views}
                </span>
              </div>
            </div>
          );
        })}
      </div>

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
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center">
                    <MapPin className="mr-1 size-3.5" />
                    {zoomed.location}
                  </span>
                  <span className="flex items-center">
                    <Eye className="mr-1 size-3.5" />
                    {zoomed.views}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
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
                    {liked.includes(zoomed.id) ? "Liked" : "Like"}
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
