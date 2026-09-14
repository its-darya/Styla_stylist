import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bookmark, Check, Layers, Shirt, Shuffle, Sparkles, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { OutfitCard } from "@/components/styla/OutfitCard";
import { ItemPicker } from "@/components/styla/ItemPicker";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { UploadZone } from "@/components/styla/UploadZone";
import { TryOnPanel } from "@/components/styla/TryOnPanel";
import { useStyla } from "@/lib/styla/store";
import {
  clearPersonalStyle,
  generateOutfit,
  getPersonalStyleCount,
  uploadPersonalStyleRef,
} from "@/lib/styla/api";
import { STYLES, type Outfit, type OuterwearMode, type StyleId } from "@/lib/styla/types";
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

const BATCH = 6;

function GeneratePage() {
  const { wardrobe, savedLooks, saveLook } = useStyla();

  const [style, setStyle] = useState<StyleId>("casual");
  const [gender, setGender] = useState("any");
  const [outerwear, setOuterwear] = useState<OuterwearMode>("auto");
  const [anchorId, setAnchorId] = useState<string | null>(null);
  const [usePersonalStyle, setUsePersonalStyle] = useState(false);
  const [personalCount, setPersonalCount] = useState(0);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const [outfits, setOutfits] = useState<Outfit[]>([]);
  /** The settings the visible results were produced with, so the heading
      describes those looks rather than the controls' current state. */
  const [applied, setApplied] = useState<{ anchorId: string | null; style: StyleId } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** The look whose "Try this on" button was pressed, so the panel starts
      immediately for it rather than waiting for a second click. */
  const [autoStartId, setAutoStartId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState<string | null>(null);
  const offsetRef = useRef(0);
  const resultsRef = useRef<HTMLDivElement>(null);
  const tryOnRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getPersonalStyleCount().then(setPersonalCount).catch(() => {});
  }, []);

  const anchor = useMemo(
    () => wardrobe.find((i) => i.id === anchorId) ?? null,
    [wardrobe, anchorId],
  );

  const hasTop = wardrobe.some((i) => i.category === "top");
  const hasBottom = wardrobe.some((i) => i.category === "bottom");
  const hasDress = wardrobe.some((i) => i.category === "dress");
  const canGenerate = (hasTop && hasBottom) || hasDress;

  // Changing any input invalidates the current page of results.
  useEffect(() => {
    offsetRef.current = 0;
  }, [style, gender, outerwear, anchorId, usePersonalStyle]);

  const run = useCallback(
    async (append: boolean) => {
      if (!wardrobe.length) {
        toast.error("Add a few garments to your wardrobe first");
        return;
      }
      if (!canGenerate) {
        toast.error("You need a top and a bottom, or a dress, to build an outfit");
        return;
      }
      if (usePersonalStyle && !referenceFile && personalCount === 0) {
        toast.error("Upload a photo of a look you like so Styla can learn your taste");
        return;
      }

      setLoading(true);
      if (!append) {
        setOutfits([]);
        setSelectedId(null);
        offsetRef.current = 0;
      }
      try {
        let refs = personalCount;
        if (usePersonalStyle && referenceFile) {
          try {
            refs = await uploadPersonalStyleRef(referenceFile);
            setPersonalCount(refs);
            setReferenceFile(null);
            setPreview(null);
            toast.success(`Style reference saved (${refs} total)`);
          } catch (err) {
            toast.error((err as Error).message || "Failed to process the reference photo");
            return;
          }
        }

        const options = {
          gender,
          usePersonalStyle: usePersonalStyle && refs > 0,
          count: BATCH,
          outerwear,
          offset: append ? offsetRef.current : 0,
          ...(anchorId ? { mustInclude: anchorId } : {}),
        };
        const results = await generateOutfit(style, options);

        if (results.length === 0) {
          toast.error(
            append
              ? "That's every combination for these settings"
              : "No outfit fits these settings yet. Try another style or clear the filters.",
          );
          return;
        }
        offsetRef.current = (append ? offsetRef.current : 0) + results.length;
        setOutfits((prev) => (append ? [...prev, ...results] : results));
        setApplied({ anchorId, style });
        setAutoStartId(null);
        if (!append) {
          setSelectedId(results[0]?.id ?? null);
          requestAnimationFrame(() =>
            resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
          );
        }
      } catch (err) {
        toast.error((err as Error).message || "Couldn't generate outfits");
      } finally {
        setLoading(false);
      }
    },
    [
      wardrobe.length, canGenerate, usePersonalStyle, referenceFile, personalCount,
      gender, outerwear, anchorId, style,
    ],
  );

  function tryThisOn(outfit: Outfit) {
    setSelectedId(outfit.id);
    setAutoStartId(outfit.id);
    // Bring the panel into view, otherwise picking a look further down the
    // list appears to do nothing.
    requestAnimationFrame(() =>
      tryOnRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }),
    );
  }

  async function onSave(outfit: Outfit) {
    setSavingId(outfit.id);
    try {
      await saveLook(outfit);
      toast.success("Saved to your looks");
    } catch (err) {
      toast.error((err as Error).message || "Couldn't save this look");
    } finally {
      setSavingId(null);
    }
  }

  async function forgetPersonalStyle() {
    try {
      await clearPersonalStyle();
      setPersonalCount(0);
      setReferenceFile(null);
      setPreview(null);
      toast.success("Cleared your style references");
    } catch (err) {
      toast.error((err as Error).message || "Couldn't clear references");
    }
  }

  const selected = outfits.find((o) => o.id === selectedId) ?? outfits[0] ?? null;
  const appliedAnchor = applied?.anchorId
    ? wardrobe.find((i) => i.id === applied.anchorId) ?? null
    : null;
  const stale =
    outfits.length > 0 && applied !== null &&
    (applied.anchorId !== anchorId || applied.style !== style);

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Generate</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Pick a mood. Get a look.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Every combination is built only from pieces already hanging in your wardrobe, scored for
          colour harmony, style fit and — if you like — your own taste.
        </p>
      </header>

      {/* Controls */}
      <section className="glass space-y-4 rounded-3xl p-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="space-y-1.5">
            <Label className="text-[11px] uppercase tracking-wider text-muted-foreground">Style</Label>
            <Select value={style} onValueChange={(v) => setStyle(v as StyleId)}>
              <SelectTrigger className="w-full rounded-xl bg-background/60">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STYLES.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    <div className="flex items-center">
                      <span
                        className="mr-2 inline-block size-3 rounded-full"
                        style={{ backgroundColor: s.tint }}
                      />
                      {s.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-[11px] uppercase tracking-wider text-muted-foreground">Fit for</Label>
            <Select value={gender} onValueChange={setGender}>
              <SelectTrigger className="w-full rounded-xl bg-background/60">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Anyone</SelectItem>
                <SelectItem value="menswear">Menswear</SelectItem>
                <SelectItem value="womenswear">Womenswear</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Jackets &amp; coats
            </Label>
            <Select value={outerwear} onValueChange={(v) => setOuterwear(v as OuterwearMode)}>
              <SelectTrigger className="w-full rounded-xl bg-background/60">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Add when it suits</SelectItem>
                <SelectItem value="always">Always add one</SelectItem>
                <SelectItem value="never">Never add one</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Build around
            </Label>
            <ItemPicker
              wardrobe={wardrobe}
              selectedId={anchorId}
              onSelect={setAnchorId}
              trigger={
                <Button
                  variant="outline"
                  className="h-9 w-full justify-start gap-2 rounded-xl bg-background/60 px-2 font-normal"
                >
                  {anchor ? (
                    <>
                      <img
                        src={anchor.thumbnailUrl ?? anchor.imageUrl}
                        alt=""
                        className="size-6 shrink-0 rounded bg-white object-contain"
                      />
                      <span className="truncate">
                        {anchor.color} {anchor.fineCategory || anchor.category}
                      </span>
                    </>
                  ) : (
                    <>
                      <Layers className="size-4 shrink-0 text-muted-foreground" />
                      <span className="text-muted-foreground">Any piece</span>
                    </>
                  )}
                </Button>
              }
            />
          </div>
        </div>

        {/* Personal style */}
        <div className="flex flex-wrap items-center gap-3 border-t border-border/50 pt-4">
          <div className="flex items-center gap-2">
            <Switch
              id="personal-style"
              checked={usePersonalStyle}
              onCheckedChange={setUsePersonalStyle}
            />
            <Label htmlFor="personal-style">Match my personal style</Label>
          </div>
          {personalCount > 0 && (
            <>
              <span className="rounded-full bg-accent-soft px-3 py-1 text-xs text-primary">
                {personalCount} reference photo{personalCount === 1 ? "" : "s"}
              </span>
              <Button variant="ghost" size="sm" className="rounded-full" onClick={forgetPersonalStyle}>
                <Trash2 className="size-3.5" /> Forget
              </Button>
            </>
          )}
          {anchor && (
            <Button
              variant="ghost"
              size="sm"
              className="rounded-full"
              onClick={() => setAnchorId(null)}
            >
              <X className="size-3.5" /> Clear build-around
            </Button>
          )}
        </div>

        {usePersonalStyle && (
          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
            <UploadZone
              title={personalCount ? "Add another reference" : "Upload a look you love"}
              subtitle="A photo of an outfit that feels like you — Styla learns from it"
              onFile={(file) => {
                setReferenceFile(file);
                setPreview(URL.createObjectURL(file));
              }}
              disabled={loading}
              className="py-8"
            />
            {preview && (
              <img
                src={preview}
                alt="Style reference"
                className="glass h-full max-h-48 w-full rounded-3xl object-cover p-1.5"
              />
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2 border-t border-border/50 pt-4">
          <Button onClick={() => run(false)} disabled={loading} className="rounded-full px-6">
            <Sparkles className="size-4" />
            {loading ? "Styling…" : outfits.length ? "Generate again" : "Generate outfits"}
          </Button>
          {outfits.length > 0 && (
            <Button
              variant="outline"
              className="rounded-full"
              onClick={() => run(true)}
              disabled={loading}
            >
              <Shuffle className="size-4" /> Show me more
            </Button>
          )}
        </div>
      </section>

      {loading && outfits.length === 0 && (
        <AnalyzingCard
          steps={[
            "Reading colour harmony across your wardrobe…",
            "Filtering pieces that fit the chosen style…",
            "Scoring compatibility and layers…",
            "Assembling your looks…",
          ]}
        />
      )}

      {outfits.length > 0 && (
        <div ref={resultsRef} className="space-y-4">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-primary">Your looks</p>
              <h2 className="mt-1 text-2xl">
                {outfits.length} outfit{outfits.length === 1 ? "" : "s"}
                {appliedAnchor
                  ? ` with your ${appliedAnchor.fineCategory || appliedAnchor.category}`
                  : ""}
              </h2>
            </div>
            {stale ? (
              <p className="text-xs font-medium text-primary">
                Settings changed — generate again
              </p>
            ) : (
              <p className="hidden text-xs text-muted-foreground sm:block">
                Pick a look, then try it on
              </p>
            )}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {outfits.map((outfit) => {
              const saved = savedLooks.some((o) => o.id === outfit.id);
              return (
                <OutfitCard
                  key={outfit.id}
                  outfit={outfit}
                  selected={outfit.id === selected?.id}
                  onSelect={() => {
                    setSelectedId(outfit.id);
                    setAutoStartId(null);
                  }}
                  showBreakdown
                  action={
                    <Button
                      variant={saved ? "secondary" : "ghost"}
                      size="sm"
                      className="rounded-full"
                      disabled={saved || savingId === outfit.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        void onSave(outfit);
                      }}
                    >
                      {saved ? <Check className="size-4" /> : <Bookmark className="size-4" />}
                      {saved ? "Saved" : savingId === outfit.id ? "Saving…" : "Save"}
                    </Button>
                  }
                  footer={
                    <Button
                      variant={outfit.id === selected?.id ? "default" : "outline"}
                      size="sm"
                      className="rounded-full"
                      onClick={(e) => {
                        e.stopPropagation();
                        tryThisOn(outfit);
                      }}
                    >
                      <Shirt className="size-4" />
                      {outfit.id === selected?.id ? "Trying this on" : "Try this on"}
                    </Button>
                  }
                />
              );
            })}
          </div>

          {loading && (
            <p className="text-center text-sm text-muted-foreground">Finding more looks…</p>
          )}

          {selected && (
            <div ref={tryOnRef} className={cn("space-y-3 pt-2", loading && "opacity-60")}>
              <p className="text-xs uppercase tracking-[0.25em] text-primary">
                Trying on: {selected.summary ?? "selected look"}
              </p>
              <TryOnPanel
                key={selected.id}
                outfitId={selected.id}
                items={selected.items}
                autoStart={selected.id === autoStartId}
              />
            </div>
          )}
        </div>
      )}

      {!loading && outfits.length === 0 && (
        <div className="glass rounded-3xl px-6 py-16 text-center">
          <h2 className="font-display text-2xl">No looks yet</h2>
          <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
            {canGenerate
              ? "Choose a style above and hit generate."
              : "Add at least one top and one bottom, or a dress, to start styling."}
          </p>
        </div>
      )}
    </div>
  );
}
