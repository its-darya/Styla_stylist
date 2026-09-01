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
      <div className="flex flex-col items-center justify-center -space-y-8 py-8 bg-white/50 rounded-2xl">
        {[...outfit.items]
          .sort((a, b) => {
            const order: Record<string, number> = {
              hat: 1, sunglasses: 2, scarf: 3, outerwear: 4, coat: 4, jacket: 5, 
              top: 7, sweater: 6, shirt: 7, 't-shirt': 8, dress: 9, 
              belt: 10, bottom: 12, pants: 11, jeans: 12, skirt: 13, shorts: 14, 
              bag: 15, shoes: 16, boots: 17
            };
            return (order[a.category] || 99) - (order[b.category] || 99);
          })
          .map((item, i) => (
          <figure key={item.id} className="relative z-10 w-48 transition-transform hover:scale-110 hover:z-20" style={{ zIndex: 10 - i }}>
            <img
              src={item.imageUrl}
              alt={categoryLabel(item.category)}
              loading="lazy"
              className="w-full object-contain drop-shadow-xl mix-blend-multiply"
            />
          </figure>
        ))}
      </div>
    </article>
  );
}
