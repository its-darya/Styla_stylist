import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { AnalyzingCard, ItemSkeleton } from "@/components/styla/Analyzing";
import { ItemCard } from "@/components/styla/ItemCard";
import { UploadZone } from "@/components/styla/UploadZone";
import { useStyla } from "@/lib/styla/store";
import { analyzeGarmentPhoto } from "@/lib/styla/mock-api";
import { CATEGORIES, type Category } from "@/lib/styla/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Styla — Your AI Personal Stylist Wardrobe" },
      {
        name: "description",
        content:
          "Build a digital wardrobe from photos of your own clothes and let Styla style them into outfits.",
      },
      { property: "og:title", content: "Styla — Your AI Personal Stylist Wardrobe" },
      {
        property: "og:description",
        content: "Digitise your closet, generate outfits, and recreate looks you love.",
      },
    ],
  }),
  component: WardrobePage,
});

type Filter = Category | "all";

function WardrobePage() {
  const { wardrobe, loadingWardrobe, addItem, removeItem } = useStyla();
  const [filter, setFilter] = useState<Filter>("all");
  const [analyzing, setAnalyzing] = useState(false);

  const visible = useMemo(
    () => (filter === "all" ? wardrobe : wardrobe.filter((i) => i.category === filter)),
    [wardrobe, filter],
  );

  async function handleFile(file: File) {
    setAnalyzing(true);
    try {
      const item = await analyzeGarmentPhoto(file);
      addItem(item);
      toast.success(`Added a ${item.color.toLowerCase()} ${item.category}`);
    } finally {
      setAnalyzing(false);
    }
  }

  const chips: { id: Filter; label: string }[] = [
    { id: "all", label: "All" },
    ...CATEGORIES.map((c) => ({ id: c.id as Filter, label: c.plural })),
  ];

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Wardrobe</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Everything you own, in one place.</h1>
        <p className="mt-3 max-w-xl text-muted-foreground">
          Snap each garment once. Styla segments it, reads the colour and category, and files it
          away for styling.
        </p>
      </header>

      <UploadZone
        title={wardrobe.length ? "Add another garment" : "Upload your first garment"}
        subtitle="Drag & drop a photo, or click to browse"
        onFile={handleFile}
        disabled={analyzing}
      />

      {analyzing && (
        <AnalyzingCard
          steps={[
            "Segmenting garment from background…",
            "Classifying category…",
            "Extracting dominant colour & pattern…",
            "Filing into your wardrobe…",
          ]}
        />
      )}

      <div className="flex flex-wrap gap-2">
        {chips.map((c) => (
          <button
            key={c.id}
            onClick={() => setFilter(c.id)}
            className={cn(
              "rounded-full border border-border/70 px-4 py-1.5 text-sm transition-colors",
              filter === c.id
                ? "border-primary bg-primary text-primary-foreground"
                : "glass text-muted-foreground hover:text-foreground",
            )}
          >
            {c.label}
          </button>
        ))}
      </div>

      {loadingWardrobe ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <ItemSkeleton key={i} />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <div className="glass rounded-3xl px-6 py-16 text-center">
          <h2 className="font-display text-2xl">Nothing here yet</h2>
          <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
            {wardrobe.length
              ? "No pieces in this category — try another filter or add a new photo."
              : "Upload a photo of a piece you actually wear. Two or three is enough to start styling."}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {visible.map((item) => (
            <ItemCard key={item.id} item={item} onDelete={removeItem} />
          ))}
        </div>
      )}
    </div>
  );
}
