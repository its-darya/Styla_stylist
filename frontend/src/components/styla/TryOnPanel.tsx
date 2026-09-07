import { useState, useCallback, useMemo } from "react";
import { Shirt, Download, Loader2, Sparkles, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { WardrobeItem } from "@/lib/styla/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TryOnPanelProps {
  outfitId: string;
  items: WardrobeItem[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data:…;base64, prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

const STEPS = [
  "Uploading your photo…",
  "Detecting clothing regions…",
  "Generating try-on with Kolors AI…",
  "Compositing result…",
  "Almost there…",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TryOnPanel({ outfitId, items }: TryOnPanelProps) {
  const outfitGender = useMemo(() => {
    const hasWomen = items.some(i => i.gender?.toLowerCase() === 'women' || i.gender?.toLowerCase() === 'female');
    return hasWomen ? 'female' : 'male';
  }, [items]);

  const defaultAvatar = `http://localhost:8000/data/avatars/base_${outfitGender}.png`;
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [personPreview, setPersonPreview] = useState<string | null>(defaultAvatar);
  const [loading, setLoading] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
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

  async function runTryOn() {
    if (!items.length) {
      toast.error("No outfit items to try on.");
      return;
    }

    setLoading(true);
    setStepIdx(0);
    setResultUrl(null);

    // Animate through steps every ~3 s while we wait
    const timer = setInterval(() => {
      setStepIdx((i) => Math.min(i + 1, STEPS.length - 1));
    }, 3000);

    try {
      let b64 = "";
      if (personFile) {
        b64 = await fileToBase64(personFile);
      }

      const payload = {
        person_image_b64: b64,
        outfit_id: outfitId,
        items: items.map((item) => ({
          id: item.id,
          imageUrl: item.imageUrl,
          category: item.category,
          color: item.color,
        })),
      };

      const res = await fetch("http://localhost:8000/api/tryon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Try-on failed");
      }

      const data = await res.json();
      setResultUrl(data.result_url);
      toast.success("Your virtual try-on is ready! 🎉");
    } catch (err: unknown) {
      toast.error((err as Error).message ?? "Try-on failed. Please try again.");
    } finally {
      clearInterval(timer);
      setLoading(false);
      setStepIdx(0);
    }
  }

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
            Powered by Kolors AI · Upload your photo to see how this outfit looks on you
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
                Try it on with Kolors AI
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
