import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { Shirt, Download, Loader2, Sparkles, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { API_BASE, startTryOn } from "@/lib/styla/api";
import type { WardrobeItem } from "@/lib/styla/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TryOnPanelProps {
  outfitId: string;
  items: WardrobeItem[];
  /** Start the try-on as soon as the panel appears, because the user asked
      for this specific look rather than merely selecting it. */
  autoStart?: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STEPS = [
  "Uploading your photo…",
  "Detecting clothing regions…",
  "Generating the try-on…",
  "Compositing result…",
  "Almost there…",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TryOnPanel({ outfitId, items, autoStart = false }: TryOnPanelProps) {
  const outfitGender = useMemo(() => {
    const hasWomen = items.some((i) => {
      const g = i.gender?.toLowerCase() ?? "";
      return g.includes("women") || g.includes("female");
    });
    return hasWomen ? 'female' : 'male';
  }, [items]);

  const defaultAvatar = `${API_BASE}/data/avatars/base_${outfitGender}.png`;
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [personPreview, setPersonPreview] = useState<string | null>(defaultAvatar);
  const [loading, setLoading] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [mode, setMode] = useState<"ai" | "preview">("ai");
  const [dragOver, setDragOver] = useState(false);

  // ---- file handling -------------------------------------------------------

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file.");
      return;
    }
    setPersonFile(file);
    setPersonPreview(URL.createObjectURL(file));
    setResultUrl(null);
  }, []);

  const clearPerson = useCallback(() => {
    setPersonFile(null);
    setPersonPreview(defaultAvatar);
    setResultUrl(null);
  }, [defaultAvatar]);

  // ---- try-on --------------------------------------------------------------

  const runRef = useRef<() => void>(() => {});

  // Fire once per mounted outfit. The parent remounts this panel (keyed on
  // the outfit id) when a different look is chosen, so this runs for exactly
  // the look the user pressed the button on.
  useEffect(() => {
    if (autoStart) runRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, outfitId]);

  async function runTryOn() {
    if (!items.length) {
      toast.error("No outfit items to try on.");
      return;
    }

    setLoading(true);
    setStepIdx(0);
    setResultUrl(null);
    setSkipped([]);
    setMode("ai");

    // Animate through steps every ~3 s while we wait
    const timer = setInterval(() => {
      setStepIdx((i) => Math.min(i + 1, STEPS.length - 1));
    }, 3000);

    try {
      const data = await startTryOn(outfitId, items, personFile);
      setResultUrl(data.result_url ?? null);
      setSkipped(data.skipped ?? []);
      setMode(data.mode ?? "ai");
      if (data.skipped && data.skipped.length > 0) {
        // Be honest about a partial result rather than showing an image that
        // quietly omits a garment.
        toast.warning(`Couldn't render: ${data.skipped.join(", ")}`);
      } else if (data.mode === "preview") {
        toast.success("Quick preview ready");
      } else {
        toast.success("Your virtual try-on is ready!");
      }
    } catch (err: unknown) {
      toast.error((err as Error).message ?? "Try-on failed. Please try again.");
    } finally {
      clearInterval(timer);
      setLoading(false);
      setStepIdx(0);
    }
  }

  runRef.current = () => {
    if (!loading) void runTryOn();
  };

  // ---- render --------------------------------------------------------------

  return (
    <section className="glass rounded-3xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border/40 px-6 py-4">
        <span className="grid size-9 place-items-center rounded-2xl bg-accent-soft text-primary">
          <Shirt className="size-4" />
        </span>
        <div>
          <p className="font-display text-lg leading-none">Virtual Try-On</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Upload your photo to see how this outfit looks on you
          </p>
        </div>
      </div>

      <div className="grid gap-6 p-6 md:grid-cols-2">
        {/* Left: person upload */}
        <div className="flex flex-col gap-3">
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
            Your photo
          </p>

          {personPreview ? (
            <div className="relative aspect-[3/4] overflow-hidden rounded-2xl bg-muted">
              <img
                src={personPreview}
                alt="Person preview"
                className="size-full object-cover"
              />
              {/* clear button */}
              {personFile && (
              <button
                onClick={clearPerson}
                className="absolute right-2 top-2 grid size-7 place-items-center rounded-full bg-black/60 text-white backdrop-blur transition-colors hover:bg-black/80"
                aria-label="Remove photo"
                id="tryon-clear-person-btn"
              >
                <X className="size-3.5" />
              </button>
              )}
            </div>
          ) : (
            /* Drop Zone */
            <div
              id="tryon-person-dropzone"
              role="button"
              tabIndex={0}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer.files?.[0];
                if (f) handleFile(f);
              }}
              onClick={() => document.getElementById("tryon-file-input")?.click()}
              onKeyDown={(e) => e.key === "Enter" && document.getElementById("tryon-file-input")?.click()}
              className={cn(
                "flex aspect-[3/4] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-border/60 text-center transition-all",
                dragOver && "border-primary bg-accent-soft/50",
                loading && "pointer-events-none opacity-60",
              )}
            >
              <span className="grid size-12 place-items-center rounded-2xl bg-accent-soft text-primary">
                <UploadCloud className="size-5" />
              </span>
              <div>
                <p className="font-display text-base">Drop your photo here</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  JPEG or PNG · any size
                </p>
              </div>
            </div>
          )}

          <input
            id="tryon-file-input"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
              e.target.value = "";
            }}
          />

          <Button
            id="tryon-run-btn"
            onClick={runTryOn}
            disabled={loading}
            className="rounded-full w-full"
          >
            {loading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                {STEPS[stepIdx]}
              </>
            ) : (
              <>
                <Sparkles className="size-4" />
                Try it on
              </>
            )}
          </Button>
        </div>

        {/* Right: result */}
        <div className="flex flex-col gap-3">
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
            Result
          </p>

          {loading && (
            <div className="flex aspect-[3/4] flex-col items-center justify-center gap-4 rounded-2xl bg-accent-soft/30">
              <div className="relative size-16">
                <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
                <div className="absolute inset-0 animate-spin rounded-full border-4 border-transparent border-t-primary" />
                <Sparkles className="absolute inset-0 m-auto size-6 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground animate-pulse">
                {STEPS[stepIdx]}
              </p>
            </div>
          )}

          {!loading && resultUrl && (
            <div className="relative aspect-[3/4] overflow-hidden rounded-2xl bg-muted animate-in fade-in slide-in-from-bottom-4 duration-500">
              <img
                src={resultUrl}
                alt="Virtual try-on result"
                className="size-full object-cover"
              />
              {/* Download button */}
              <a
                id="tryon-download-btn"
                href={resultUrl}
                download="styla-tryon.png"
                target="_blank"
                rel="noreferrer"
                className="absolute bottom-3 right-3 inline-flex items-center gap-1.5 rounded-full bg-black/70 px-3 py-1.5 text-xs font-medium text-white backdrop-blur transition-colors hover:bg-black/90"
              >
                <Download className="size-3.5" />
                Save photo
              </a>
            </div>
          )}

          {!loading && resultUrl && mode === "preview" && (
            <p className="rounded-xl bg-secondary/70 px-3 py-2 text-xs text-muted-foreground">
              Quick preview, drawn from your garments. The AI model was out of
              free GPU time, so this shows the pieces on the body rather than a
              photorealistic render.
            </p>
          )}

          {!loading && resultUrl && skipped.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Shown without {skipped.join(" and ")} — the try-on service could not
              render {skipped.length === 1 ? "it" : "them"} this time.
            </p>
          )}

          {!loading && !resultUrl && (
            <div className="flex aspect-[3/4] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border/40 text-center">
              <Shirt className="size-10 text-border" />
              <p className="text-sm text-muted-foreground">
                Upload your photo and hit<br />
                <span className="text-primary font-medium">Try it on</span>
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
