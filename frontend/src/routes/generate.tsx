import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Bookmark, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { OutfitCard } from "@/components/styla/OutfitCard";
import { AnalyzingCard } from "@/components/styla/Analyzing";
import { UploadZone } from "@/components/styla/UploadZone";
import { useStyla } from "@/lib/styla/store";
import { generateOutfit, uploadPersonalStyleRef } from "@/lib/styla/mock-api";
import { STYLES, type Outfit, type StyleId } from "@/lib/styla/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/generate")({
  head: () => ({
    meta: [
      { title: "Generate Outfits â€” Styla" },
      {
        name: "description",
        content: "Pick a style and let Styla combine pieces from your own wardrobe into a look.",
      },
      { property: "og:title", content: "Generate Outfits â€” Styla" },
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
  const [gender, setGender] = useState<string>("any");
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<Outfit[]>([]);
  const [outfitIndex, setOutfitIndex] = useState(0);
  const [usePersonalStyle, setUsePersonalStyle] = useState(false);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  async function run() {
    if (!wardrobe.length) {
      toast.error("Add a few garments to your wardrobe first");
      return;
    }
    if (usePersonalStyle && !referenceFile) {
      toast.error("Please upload a reference image for your personal style");
      return;
    }
    setLoading(true);
    setOutfits([]);
    setOutfitIndex(0);
    try {
      const userId = "user123";
      if (usePersonalStyle && referenceFile) {
        const success = await uploadPersonalStyleRef(referenceFile, userId);
        if (!success) {
          toast.error("Failed to process reference image");
          return;
        }
      }
      const results = await generateOutfit(style, wardrobe, usePersonalStyle ? userId : undefined);
      setOutfits(results);
    } finally {
      setLoading(false);
    }
  }
  
  function nextOutfit() {
    if (outfits.length > 0) {
      setOutfitIndex((i) => (i + 1) % outfits.length);
    }
  }

  function handleFile(file: File) {
    setReferenceFile(file);
    setPreview(URL.createObjectURL(file));
  }

  const currentOutfit = outfits[outfitIndex];
  const saved = currentOutfit ? savedLooks.some((o) => o.id === currentOutfit.id) : false;

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Generate</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Pick a mood. Get a look.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Every combination is built only from pieces already hanging in your wardrobe.
        </p>
      </header>



      <div className="flex flex-wrap gap-3">
        <div className="flex items-center space-x-2 w-full mb-2">
          <Switch id="personal-style" checked={usePersonalStyle} onCheckedChange={setUsePersonalStyle} />
          <Label htmlFor="personal-style">Mənim öz stilimə uyğun et</Label>
        </div>
        
        {usePersonalStyle && (
          <div className="grid gap-4 w-full md:grid-cols-[minmax(0,1fr)_16rem] mb-4">
            <UploadZone
              title="Upload your style reference"
              subtitle="Drop a photo you like"
              onFile={handleFile}
              disabled={loading}
            />
            {preview && (
              <img
                src={preview}
                alt="Reference style"
                className="glass h-full max-h-48 w-full rounded-3xl object-cover p-1.5"
              />
            )}
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2">
          <Select value={gender} onValueChange={setGender}>
            <SelectTrigger className="w-[120px] rounded-full glass border-primary/20 bg-background/50">
              <SelectValue placeholder="Gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="any">H?r ikisi</SelectItem>
              <SelectItem value="men">Kisi</SelectItem>
              <SelectItem value="women">Qadin</SelectItem>
            </SelectContent>
          </Select>

          <Select value={style} onValueChange={(v) => setStyle(v as StyleId)}>
            <SelectTrigger className="w-[180px] rounded-full glass border-primary/20 bg-background/50">
              <SelectValue placeholder="Select style" />
            </SelectTrigger>
            <SelectContent>
              {STYLES.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  <div className="flex items-center">
                    <span className="mr-2 inline-block size-3 rounded-full" style={{ backgroundColor: s.tint }} />
                    {s.label}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button onClick={run} disabled={loading} className="rounded-full px-6">
            <Sparkles className="size-4" />
            {loading ? "Stylingâ€¦" : "Generate outfit"}
          </Button>
        </div>
        {currentOutfit && (
          <>
            <Button variant="outline" className="rounded-full" onClick={nextOutfit} disabled={loading || outfits.length <= 1}>
              <RefreshCw className="size-4" /> Regenerate
            </Button>
            <Button
              variant="secondary"
              className="rounded-full"
              disabled={saved}
              onClick={() => {
                saveLook(currentOutfit);
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
            "Reading colour harmony across your wardrobeâ€¦",
            "Filtering pieces that fit the chosen styleâ€¦",
            "Balancing silhouette and layersâ€¦",
            "Assembling your lookâ€¦",
          ]}
        />
      )}

      {currentOutfit && !loading && <OutfitCard outfit={currentOutfit} />}

      {!currentOutfit && !loading && (
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

