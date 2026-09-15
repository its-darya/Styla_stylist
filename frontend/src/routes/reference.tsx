import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, ExternalLink, ShoppingBag } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { UploadZone } from "@/components/styla/UploadZone";
import { categoryLabel } from "@/components/styla/ItemCard";
import { useStyla } from "@/lib/styla/store";
import { matchReferenceImage } from "@/lib/styla/mock-api";
import type { ReferenceMatchResult, SuggestedProduct } from "@/lib/styla/types";

export const Route = createFileRoute("/reference")({
  head: () => ({
    meta: [
      { title: "Reference Match — Styla" },
      {
        name: "description",
        content:
          "Upload an outfit you love, match it to your wardrobe, and open the real product page on Google.",
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

function openPage(url: string) {
  window.open(url, "_blank", "noopener,noreferrer");
}

function ProductCard({ product }: { product: SuggestedProduct }) {
  return (
    <button
      type="button"
      onClick={() => openPage(product.url)}
      className="rounded-2xl bg-secondary/60 p-2 text-left transition hover:-translate-y-0.5"
    >
      {product.imageUrl ? (
        <img
          src={product.imageUrl}
          alt={product.name}
          loading="lazy"
          className="aspect-square w-full rounded-xl object-cover"
        />
      ) : (
        <div className="flex aspect-square w-full items-center justify-center rounded-xl bg-muted">
          <ShoppingBag className="size-8 text-muted-foreground" />
        </div>
      )}
      <p className="mt-2 truncate text-sm font-medium">{product.name}</p>
      <div className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
        <span className="truncate text-sm text-muted-foreground">{product.price || "Shop"}</span>
        <span className="inline-flex items-center gap-1 text-xs font-medium text-primary">
          Open <ExternalLink className="size-3" />
        </span>
      </div>
    </button>
  );
}

function ReferencePage() {
  const { wardrobe } = useStyla();
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReferenceMatchResult | null>(null);

  async function handleFile(file: File) {
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);
    try {
      const next = await matchReferenceImage(file, wardrobe);
      setResult(next);
      if (next.bestUrl) {
        openPage(next.bestUrl);
        toast.success("Opening the closest match we found online");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not search this garment");
    } finally {
      setLoading(false);
    }
  }

  const shopProducts = result?.onlineProducts?.length
    ? result.onlineProducts
    : result?.missingItems.flatMap((item) => item.suggestedProducts) ?? [];

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Reference match</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Find the piece, then recreate the look.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Drop in a photo. Styla identifies the garment, matches what you already own, and opens
          the closest store page we can find on Google.
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
            "Detecting the garment…",
            "Matching it against your wardrobe…",
            "Searching Google for this piece…",
            "Opening the closest store page…",
          ]}
        />
      )}

      {result && !loading && (
        <div className="space-y-10">
          {(result.bestUrl || result.googleShoppingUrl) && (
            <section className="glass flex flex-wrap items-center justify-between gap-4 rounded-3xl p-5">
              <div className="min-w-0">
                <p className="text-xs uppercase tracking-[0.2em] text-primary">Found online</p>
                <p className="mt-1 font-display text-2xl">
                  {result.detected
                    ? `${result.detected.color} ${result.detected.category}`
                    : "This garment"}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Search: {result.query || "similar clothing"}
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                {result.bestUrl && (
                  <Button className="rounded-full" onClick={() => openPage(result.bestUrl!)}>
                    Open product page <ExternalLink className="size-3.5" />
                  </Button>
                )}
                {result.googleShoppingUrl && (
                  <Button
                    variant="outline"
                    className="rounded-full"
                    onClick={() => openPage(result.googleShoppingUrl!)}
                  >
                    Google Shopping
                  </Button>
                )}
              </div>
            </section>
          )}

          {result.matchedItems.length > 0 && (
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
          )}

          {result.missingItems.length > 0 && (
            <section className="space-y-4">
              <h2 className="text-2xl">Missing from your wardrobe</h2>
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
                        Nothing close in your wardrobe — tap a result to open the store page.
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </section>
          )}

          {shopProducts.length > 0 && (
            <section className="space-y-4">
              <h2 className="text-2xl">Shop this piece</h2>
              <div className="grid gap-3 sm:grid-cols-3">
                {shopProducts.map((p, i) => (
                  <ProductCard key={`${p.url}-${i}`} product={p} />
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
