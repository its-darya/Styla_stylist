import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { UploadZone } from "@/components/styla/UploadZone";
import { categoryLabel } from "@/components/styla/ItemCard";
import { useStyla } from "@/lib/styla/store";
import { matchReferenceImage } from "@/lib/styla/mock-api";
import type { ReferenceMatchResult } from "@/lib/styla/types";

export const Route = createFileRoute("/reference")({
  head: () => ({
    meta: [
      { title: "Reference Match — Styla" },
      {
        name: "description",
        content:
          "Upload an outfit you love and see how much of it you can recreate from your own wardrobe.",
      },
      { property: "og:title", content: "Reference Match — Styla" },
      {
        property: "og:description",
        content: "Match a saved inspiration look against your closet and shop only what's missing.",
      },
    ],
  }),
  component: ReferencePage,
});

function ReferencePage() {
  const { wardrobe } = useStyla();
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReferenceMatchResult | null>(null);

  async function handleFile(file: File) {
    if (!wardrobe.length) {
      toast.error("Add garments to your wardrobe first");
      return;
    }
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);
    try {
      setResult(await matchReferenceImage(file, wardrobe));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Reference match</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Recreate a look you saved.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Drop in a Pinterest outfit. Styla breaks it into pieces, matches what you already own, and
          suggests the rest.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_16rem]">
        <UploadZone
          title="Upload a reference outfit"
          subtitle="Screenshot, Pinterest save, or street style photo"
          onFile={handleFile}
          disabled={loading}
        />
        {preview && (
          <img
            src={preview}
            alt="Reference outfit"
            className="glass h-full max-h-72 w-full rounded-3xl object-cover p-1.5"
          />
        )}
      </div>

      {loading && (
        <AnalyzingCard
          steps={[
            "Detecting garments in the reference…",
            "Embedding each piece…",
            "Searching your wardrobe for matches…",
            "Sourcing suggestions for gaps…",
          ]}
        />
      )}

      {result && !loading && (
        <div className="space-y-10">
          <section className="space-y-4">
            <h2 className="text-2xl">In your wardrobe</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {result.matchedItems.map((m, i) => (
                <div key={i} className="glass rounded-3xl p-4">
                  <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                    <img
                      src={m.referenceImageUrl}
                      alt="Reference piece"
                      className="aspect-[3/4] w-full rounded-2xl object-cover"
                    />
                    <ArrowRight className="size-4 text-primary" />
                    <img
                      src={m.wardrobeItem.imageUrl}
                      alt={categoryLabel(m.wardrobeItem.category)}
                      className="aspect-[3/4] w-full rounded-2xl object-cover"
                    />
                  </div>
                  <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                    <p className="truncate text-sm">
                      {m.wardrobeItem.color} {categoryLabel(m.wardrobeItem.category)}
                    </p>
                    <span className="shrink-0 rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-primary">
                      {m.matchScore}% match
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="space-y-4">
            <h2 className="text-2xl">Missing pieces</h2>
            {result.missingItems.map((miss, i) => (
              <div key={i} className="glass rounded-3xl p-4">
                <div className="flex items-center gap-3">
                  <img
                    src={miss.referenceImageUrl}
                    alt={categoryLabel(miss.category)}
                    className="size-20 shrink-0 rounded-2xl object-cover"
                  />
                  <div className="min-w-0">
                    <p className="font-display text-xl">{categoryLabel(miss.category)}</p>
                    <p className="text-sm text-muted-foreground">
                      Nothing close in your wardrobe — here's what would work.
                    </p>
                  </div>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  {miss.suggestedProducts.map((p) => (
                    <div key={p.name} className="rounded-2xl bg-secondary/60 p-2">
                      <img
                        src={p.imageUrl}
                        alt={p.name}
                        loading="lazy"
                        className="aspect-square w-full rounded-xl object-cover"
                      />
                      <p className="mt-2 truncate text-sm font-medium">{p.name}</p>
                      <div className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
                        <span className="text-sm text-muted-foreground">{p.price}</span>
                        <Button asChild size="sm" variant="ghost" className="rounded-full">
                          <a href={p.url} target="_blank" rel="noreferrer">
                            View <ExternalLink className="size-3" />
                          </a>
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </section>
        </div>
      )}
    </div>
  );
}
