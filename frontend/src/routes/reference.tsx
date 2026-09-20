import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, ExternalLink, Link2, Maximize2, ShoppingBag } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { UploadZone } from "@/components/styla/UploadZone";
import { categoryLabel } from "@/components/styla/ItemCard";
import {
  API_BASE,
  getPinterestFeed,
  matchReferenceImage,
  matchReferenceImageUrl,
} from "@/lib/styla/api";
import { LOOKS } from "@/lib/styla/looks";
import type {
  DetectedPiece,
  PinterestPin,
  ReferenceMatchResult,
  SuggestedProduct,
  WardrobeItem,
} from "@/lib/styla/types";

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
  // ?image=… lets Discover hand a look straight over to be checked.
  validateSearch: (search: Record<string, unknown>) => ({
    image: typeof search["image"] === "string" ? search["image"] : undefined,
  }),
  component: ReferencePage,
});

function titleCase(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function describe(d?: DetectedPiece) {
  if (!d) return "";
  return `${titleCase(d.color)} ${d.category}${d.pattern && d.pattern !== "Solid" ? ` · ${d.pattern}` : ""}`;
}

function itemLabel(item: WardrobeItem) {
  const fine = item.fineCategory ? titleCase(item.fineCategory) : categoryLabel(item.category);
  return `${item.color} ${fine}`;
}

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

/**
 * One piece of the reference that the wardrobe can cover: the photo's garment
 * beside the one you own. Opens full size, with a way through to the wardrobe.
 */
function MatchedCard({
  match,
  preview,
}: {
  match: ReferenceMatchResult["matchedItems"][number];
  preview: string | null;
}) {
  const item = match.wardrobeItem;
  const picture = item.thumbnailUrl ?? item.imageUrl;
  const referencePicture = match.referenceImageUrl || preview || "";

  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          className="glass rounded-3xl p-4 text-left transition hover:-translate-y-0.5 hover:shadow-lg"
        >
          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
            <img
              src={referencePicture}
              alt="Reference piece"
              className="aspect-[3/4] w-full rounded-2xl bg-white object-contain"
            />
            <ArrowRight className="size-4 shrink-0 text-primary" />
            <img
              src={picture}
              alt={itemLabel(item)}
              className="aspect-[3/4] w-full rounded-2xl bg-white object-contain"
            />
          </div>
          <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{itemLabel(item)}</p>
              {match.detected && (
                <p className="truncate text-xs text-muted-foreground">
                  Looking for: {describe(match.detected)}
                </p>
              )}
            </div>
            <span className="shrink-0 rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-primary">
              {match.matchScore}% match
            </span>
          </div>
          {match.alternates && match.alternates.length > 0 && (
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[11px] uppercase tracking-widest text-muted-foreground">
                Also close
              </span>
              {match.alternates.map((a) => (
                <div key={a.wardrobeItem.id} className="flex items-center gap-1.5">
                  <img
                    src={a.wardrobeItem.thumbnailUrl ?? a.wardrobeItem.imageUrl}
                    alt={itemLabel(a.wardrobeItem)}
                    title={itemLabel(a.wardrobeItem)}
                    className="size-10 rounded-lg bg-white object-contain"
                  />
                  <span className="text-[11px] text-muted-foreground">{a.matchScore}%</span>
                </div>
              ))}
            </div>
          )}
          <p className="mt-3 flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground">
            <Maximize2 className="size-3" /> Tap to see it bigger
          </p>
        </button>
      </DialogTrigger>

      <DialogContent className="max-h-[92vh] max-w-3xl overflow-y-auto rounded-3xl">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">{itemLabel(item)}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 sm:grid-cols-2">
          <figure>
            <img
              src={referencePicture}
              alt="Reference piece"
              className="max-h-[55vh] w-full rounded-2xl bg-white object-contain"
            />
            <figcaption className="mt-2 text-center text-xs text-muted-foreground">
              In the photo{match.detected ? `: ${describe(match.detected)}` : ""}
            </figcaption>
          </figure>
          <figure>
            <img
              src={picture}
              alt={itemLabel(item)}
              className="max-h-[55vh] w-full rounded-2xl bg-white object-contain"
            />
            <figcaption className="mt-2 text-center text-xs text-muted-foreground">
              Yours: {itemLabel(item)} · {match.matchScore}% match
            </figcaption>
          </figure>
        </div>
        <Button asChild className="rounded-full">
          <Link to="/">
            Open this in my wardrobe <ArrowRight className="size-3.5" />
          </Link>
        </Button>
      </DialogContent>
    </Dialog>
  );
}

function ReferencePage() {
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReferenceMatchResult | null>(null);
  const [pins, setPins] = useState<PinterestPin[]>([]);
  const [loadingPins, setLoadingPins] = useState(true);
  const { image: handedOver } = Route.useSearch() as { image?: string };
  const checked = useRef<string | null>(null);

  useEffect(() => {
    let alive = true;
    getPinterestFeed().then((feed) => {
      if (!alive) return;
      setPins(feed);
      setLoadingPins(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  async function performMatch(match: () => Promise<ReferenceMatchResult>) {
    setResult(null);
    setLoading(true);
    try {
      setResult(await match());
    } catch (err) {
      toast.error((err as Error).message || "Couldn't analyze that look. Try again.");
    } finally {
      setLoading(false);
    }
  }

  /** Check a picture by address: a curated look, a pin, or a Discover hand-off. */
  function checkImage(url: string) {
    setPreview(url);
    void performMatch(async () => {
      try {
        // Pictures this site serves are sent as bytes, so the API never has to
        // reach back into the frontend to download them.
        const blob = await (await fetch(url)).blob();
        const name = url.split("/").pop() || "look.jpg";
        return await matchReferenceImage(new File([blob], name, { type: blob.type || "image/jpeg" }));
      } catch {
        // A remote pin: hand the API the address instead.
        return matchReferenceImageUrl(new URL(url, window.location.origin).href);
      }
    });
  }

  // A look liked in Discover arrives as ?image=… and is checked on arrival.
  useEffect(() => {
    if (!handedOver || checked.current === handedOver) return;
    checked.current = handedOver;
    checkImage(handedOver);
    // checkImage only touches state setters, so the look is what matters here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handedOver]);

  function handleFile(file: File) {
    setPreview(URL.createObjectURL(file));
    void performMatch(() => matchReferenceImage(file));
  }

  function connectPinterest() {
    window.location.href = `${API_BASE}/api/pinterest/auth`;
  }

  const total = (result?.matchedItems.length ?? 0) + (result?.missingItems.length ?? 0);

  // Curated cut-outs by default — same white-background domain as the wardrobe
  // photos, so matching is like for like. Real pins take over once Pinterest is
  // connected; the API's own stand-in feed is marked with "fb_" ids.
  const realPins = pins.filter((p) => !p.id.startsWith("fb_"));
  const outfitChoices = realPins.length
    ? realPins.map((p) => ({ id: p.id, imageUrl: p.imageUrl, title: p.title ?? "Outfit" }))
    : LOOKS.map((l) => ({ id: l.id, imageUrl: l.imageUrl, title: `${l.style} look` }));

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Reference match</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Find the piece, then recreate the look.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Pick an outfit or upload a photo. Styla splits it into top, bottom and dress, matches
          what you already own, and finds real store pages for whatever is missing.
        </p>
      </header>

      {/* Pick an outfit — curated cut-outs, or real pins once Pinterest is connected */}
      <section className="space-y-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-primary">Inspiration</p>
            <h2 className="mt-1 text-2xl">Pick an outfit</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Tap a look to check it against your wardrobe.
            </p>
          </div>

        </div>

        {loadingPins ? (
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="aspect-[3/5] animate-pulse rounded-2xl bg-secondary/60" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {outfitChoices.map((choice) => {
              const selected = preview === choice.imageUrl;
              return (
                <button
                  key={choice.id}
                  type="button"
                  onClick={() => checkImage(choice.imageUrl)}
                  disabled={loading}
                  className="group relative aspect-[3/5] overflow-hidden rounded-2xl bg-white transition-all hover:-translate-y-0.5 hover:shadow-lg disabled:opacity-60"
                >
                  <img
                    src={choice.imageUrl}
                    alt={choice.title}
                    loading="lazy"
                    className="h-full w-full object-contain transition-transform duration-500 group-hover:scale-[1.03]"
                  />
                  {selected && (
                    <span className="absolute inset-0 rounded-2xl ring-2 ring-inset ring-primary" />
                  )}
                </button>
              );
            })}
          </div>
        )}
      </section>

      {/* Upload any clothes photo */}
      <section className="space-y-4">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-primary">Your own photo</p>
          <h2 className="mt-1 text-2xl">Upload any clothes</h2>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            A whole outfit, a single garment, a shop photo or a screenshot. Styla reads whatever is
            in the picture and looks for those pieces in your wardrobe.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_16rem]">
          <UploadZone
            title="Drop a clothes photo"
            subtitle="Outfit or single garment · JPG or PNG"
            onFile={handleFile}
            disabled={loading}
          />
          {preview && (
            <img
              src={preview}
              alt="Reference outfit"
              className="glass h-full max-h-72 w-full rounded-3xl bg-white object-contain p-1.5"
            />
          )}
        </div>
      </section>

      {loading && (
        <AnalyzingCard
          steps={[
            "Detecting garments in the reference…",
            "Matching each piece against your wardrobe…",
            "Searching the web for missing pieces…",
            "Finding the closest store pages…",
          ]}
        />
      )}

      {result && !loading && (
        <div className="space-y-10">
          {/* Summary + parsed pieces */}
          <section className="glass rounded-3xl p-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.25em] text-primary">Breakdown</p>
                <h2 className="mt-1 text-2xl">
                  {total === 0
                    ? "No garments detected"
                    : `You own ${result.matchedItems.length} of ${total} piece${total === 1 ? "" : "s"}`}
                </h2>
              </div>
              {typeof result.coverage === "number" && total > 0 && (
                <span className="rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-primary">
                  {Math.round(result.coverage * 100)}% recreatable
                </span>
              )}
            </div>
            {result.pieces && result.pieces.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-3">
                {result.pieces.map((p, i) => (
                  <div key={i} className="flex items-center gap-2 rounded-2xl bg-secondary/60 p-1.5 pr-3">
                    <img src={p.imageUrl} alt={p.slot} className="size-12 rounded-xl bg-white object-cover" />
                    <div>
                      <p className="text-xs uppercase tracking-widest text-muted-foreground">{p.slot}</p>
                      <p className="text-sm font-medium">{describe(p)}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

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

          {result.matchedItems.length > 0 ? (
            <section className="space-y-4">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 className="text-2xl">In your wardrobe</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {result.matchedItems.length === 1
                      ? "You already own this piece."
                      : `You already own ${result.matchedItems.length} of these pieces.`}{" "}
                    Tap one to see it full size.
                  </p>
                </div>
                <Button asChild variant="outline" size="sm" className="rounded-full">
                  <Link to="/">
                    Open my wardrobe <ArrowRight className="size-3.5" />
                  </Link>
                </Button>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {result.matchedItems.map((m, i) => (
                  <MatchedCard key={i} match={m} preview={preview} />
                ))}
              </div>
            </section>
          ) : (
            total > 0 && (
              <div className="glass rounded-3xl p-8 text-center">
                <p className="font-display text-2xl">Nothing close enough yet</p>
                <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
                  None of the pieces in this look are close to what you own. Check the closest
                  matches below or add more garments to your wardrobe.
                </p>
              </div>
            )
          )}

          {result.missingItems.length > 0 && (
            <section className="space-y-4">
              <h2 className="text-2xl">Missing from your wardrobe</h2>
              {result.missingItems.map((miss, i) => (
                <div key={i} className="glass rounded-3xl p-4">
                  <div className="flex flex-wrap items-center gap-4">
                    <Dialog>
                      <DialogTrigger asChild>
                        <button
                          type="button"
                          className="shrink-0 rounded-2xl bg-white p-1 transition hover:-translate-y-0.5 hover:shadow-md"
                          title="See this piece bigger"
                        >
                          <img
                            src={miss.referenceImageUrl || preview || ""}
                            alt={categoryLabel(miss.category)}
                            className="size-28 rounded-xl bg-white object-contain sm:size-40"
                          />
                        </button>
                      </DialogTrigger>
                      <DialogContent className="max-h-[92vh] max-w-xl overflow-y-auto rounded-3xl">
                        <DialogHeader>
                          <DialogTitle className="font-display text-2xl">
                            {miss.detected ? describe(miss.detected) : categoryLabel(miss.category)}
                          </DialogTitle>
                        </DialogHeader>
                        <img
                          src={miss.referenceImageUrl || preview || ""}
                          alt={categoryLabel(miss.category)}
                          className="max-h-[60vh] w-full rounded-2xl bg-white object-contain"
                        />
                        <p className="text-sm text-muted-foreground">
                          This piece isn&apos;t in your wardrobe. The shops below sell the closest
                          thing Styla could find.
                        </p>
                        {miss.googleShoppingUrl && (
                          <Button
                            className="rounded-full"
                            onClick={() => openPage(miss.googleShoppingUrl!)}
                          >
                            Find it online <ExternalLink className="size-3.5" />
                          </Button>
                        )}
                      </DialogContent>
                    </Dialog>
                    {/* A basis wide enough that the description keeps its own
                        line rather than stacking a word at a time between the
                        two pictures. */}
                    <div className="min-w-0 flex-1 basis-56">
                      <p className="text-[11px] uppercase tracking-widest text-muted-foreground">
                        Not in your wardrobe
                      </p>
                      <p className="font-display text-xl">
                        {miss.detected ? describe(miss.detected) : categoryLabel(miss.category)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {miss.closest
                          ? `Closest you own: ${itemLabel(miss.closest.wardrobeItem)} (${miss.closest.matchScore}%)`
                          : "Nothing in this category in your wardrobe yet."}
                      </p>
                    </div>
                    {miss.closest && (
                      <img
                        src={miss.closest.wardrobeItem.thumbnailUrl ?? miss.closest.wardrobeItem.imageUrl}
                        alt={itemLabel(miss.closest.wardrobeItem)}
                        title={`Closest you own: ${itemLabel(miss.closest.wardrobeItem)}`}
                        className="size-28 shrink-0 rounded-2xl bg-white object-contain opacity-80 sm:size-40"
                      />
                    )}
                  </div>
                  {miss.suggestedProducts.length > 0 && (
                    <div className="mt-4 space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="flex items-center gap-1 text-[11px] uppercase tracking-widest text-muted-foreground">
                          <ShoppingBag className="size-3.5" /> Shop this piece
                        </span>
                        {miss.googleShoppingUrl && (
                          <Button
                            size="sm"
                            variant="ghost"
                            className="rounded-full"
                            onClick={() => openPage(miss.googleShoppingUrl!)}
                          >
                            More on Google Shopping <ExternalLink className="size-3" />
                          </Button>
                        )}
                      </div>
                      <div className="grid gap-3 sm:grid-cols-3">
                        {miss.suggestedProducts.map((p, j) => (
                          <ProductCard key={`${p.url}-${j}`} product={p} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
