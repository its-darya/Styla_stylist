import { STYLES, type Outfit } from "@/lib/styla/types";
import { categoryLabel } from "./ItemCard";

export function OutfitCard({ outfit, action }: { outfit: Outfit; action?: React.ReactNode }) {
  const style = STYLES.find((s) => s.id === outfit.style);
  return (
    <article className="glass overflow-hidden rounded-3xl p-4">
      <div className="mb-3 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-display text-xl">{style?.label ?? outfit.style}</h3>
          <p className="text-xs text-muted-foreground">
            {outfit.items.length} pieces ·{" "}
            {new Date(outfit.createdAt).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
            })}
          </p>
        </div>
        {action}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {outfit.items.map((item) => (
          <figure key={item.id} className="overflow-hidden rounded-2xl bg-muted">
            <img
              src={item.imageUrl}
              alt={categoryLabel(item.category)}
              loading="lazy"
              className="aspect-[3/4] w-full object-cover"
            />
            <figcaption className="bg-surface-veil px-2 py-1.5 text-[11px] text-muted-foreground">
              {categoryLabel(item.category)} · {item.color}
            </figcaption>
          </figure>
        ))}
      </div>
    </article>
  );
}
