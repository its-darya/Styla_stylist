import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Bookmark, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { OutfitCard } from "@/components/styla/OutfitCard";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { useStyla } from "@/lib/styla/store";
import { generateOutfit } from "@/lib/styla/mock-api";
import { STYLES, type Outfit, type StyleId } from "@/lib/styla/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/generate")({
  head: () => ({
    meta: [
      { title: "Generate Outfits — Styla" },
      {
        name: "description",
        content: "Pick a style and let Styla combine pieces from your own wardrobe into a look.",
      },
      { property: "og:title", content: "Generate Outfits — Styla" },
      {
        property: "og:description",
        content: "Casual to evening: outfit combinations built only from clothes you own.",
      },
    ],
  }),
  component: GeneratePage,
});

function GeneratePage() {
  const { wardrobe, savedLooks, saveLook } = useStyla();
  const [style, setStyle] = useState<StyleId>("casual");
  const [loading, setLoading] = useState(false);
  const [outfit, setOutfit] = useState<Outfit | null>(null);

  async function run() {
    if (!wardrobe.length) {
      toast.error("Add a few garments to your wardrobe first");
      return;
    }
    setLoading(true);
    setOutfit(null);
    try {
      setOutfit(await generateOutfit(style, wardrobe));
    } finally {
      setLoading(false);
    }
  }

  const saved = outfit ? savedLooks.some((o) => o.id === outfit.id) : false;

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Generate</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Pick a mood. Get a look.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Every combination is built only from pieces already hanging in your wardrobe.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STYLES.map((s) => (
          <button
            key={s.id}
            onClick={() => setStyle(s.id)}
            className={cn(
              "glass rounded-2xl p-3 text-left transition-all hover:-translate-y-0.5",
              style === s.id && "border-primary ring-2 ring-ring",
            )}
          >
            <span
              className="mb-2 block size-6 rounded-full"
              style={{ backgroundColor: s.tint }}
              aria-hidden
            />
            <span className="block text-sm font-medium">{s.label}</span>
            <span className="block text-[11px] text-muted-foreground">{s.hint}</span>
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-3">
        <Button onClick={run} disabled={loading} className="rounded-full px-6">
          <Sparkles className="size-4" />
          {loading ? "Styling…" : "Generate outfit"}
        </Button>
        {outfit && (
          <>
            <Button variant="outline" className="rounded-full" onClick={run} disabled={loading}>
              <RefreshCw className="size-4" /> Regenerate
            </Button>
            <Button
              variant="secondary"
              className="rounded-full"
              disabled={saved}
              onClick={() => {
                saveLook(outfit);
                toast.success("Saved to your looks");
              }}
            >
              <Bookmark className="size-4" /> {saved ? "Saved" : "Save look"}
            </Button>
          </>
        )}
      </div>

      {loading && (
        <AnalyzingCard
          steps={[
            "Reading colour harmony across your wardrobe…",
            "Filtering pieces that fit the chosen style…",
            "Balancing silhouette and layers…",
            "Assembling your look…",
          ]}
        />
      )}

      {outfit && !loading && <OutfitCard outfit={outfit} />}

      {!outfit && !loading && (
        <div className="glass rounded-3xl px-6 py-16 text-center">
          <h2 className="font-display text-2xl">No look yet</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose a style above and hit generate.
          </p>
        </div>
      )}
    </div>
  );
}
