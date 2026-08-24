import { createFileRoute } from "@tanstack/react-router";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { OutfitCard } from "@/components/styla/OutfitCard";
import { useStyla } from "@/lib/styla/store";

export const Route = createFileRoute("/saved")({
  head: () => ({
    meta: [
      { title: "Saved Looks — Styla" },
      { name: "description", content: "Your favourite outfit combinations, kept in one place." },
      { property: "og:title", content: "Saved Looks — Styla" },
      {
        property: "og:description",
        content: "Revisit the outfits you saved from Styla's generator.",
      },
    ],
  }),
  component: SavedPage,
});

function SavedPage() {
  const { savedLooks, removeLook } = useStyla();

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Saved looks</p>
        <h1 className="mt-2 text-4xl md:text-5xl">Your keepers.</h1>
      </header>

      {savedLooks.length === 0 ? (
        <div className="glass rounded-3xl px-6 py-16 text-center">
          <h2 className="font-display text-2xl">No saved looks yet</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Generate an outfit and tap “Save look” to keep it here.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {savedLooks.map((o) => (
            <OutfitCard
              key={o.id}
              outfit={o}
              action={
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full"
                  onClick={() => removeLook(o.id)}
                  aria-label="Delete look"
                >
                  <Trash2 className="size-4" />
                </Button>
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
