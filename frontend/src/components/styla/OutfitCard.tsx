import type { Outfit } from "@/lib/styla/types";
import { STYLES } from "@/lib/styla/types";

function categoryLabel(category: string) {
  return category.charAt(0).toUpperCase() + category.slice(1);
}

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
      
      <div className="flex flex-row justify-center gap-4 bg-white/50 rounded-2xl p-6 min-h-[320px]">
        <div className="flex flex-wrap items-center justify-center gap-6 w-full max-w-2xl">
          {[...outfit.items]
            .sort((a, b) => {
              const order: Record<string, number> = {
                hat: 1, sunglasses: 2, scarf: 3, outerwear: 4, coat: 4, jacket: 5, 
                top: 7, sweater: 6, shirt: 7, 't-shirt': 8, dress: 9, 
                belt: 10, bottom: 12, pants: 11, jeans: 12, skirt: 13, shorts: 14, 
                bag: 15
              };
              return (order[a.category] || 99) - (order[b.category] || 99);
            })
            .map((item, i) => (
            <figure key={item.id} className="relative group w-32 md:w-44 transition-all duration-300 hover:scale-105 hover:z-20">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/5 opacity-0 group-hover:opacity-100 rounded-xl transition-opacity" />
              <img
                src={item.imageUrl}
                alt={categoryLabel(item.category)}
                loading="lazy"
                className="w-full h-44 object-contain drop-shadow-md mix-blend-multiply p-2"
              />
              <div className="absolute -bottom-2 inset-x-0 text-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground bg-white/80 backdrop-blur px-2 py-0.5 rounded-full border border-black/5">
                  {categoryLabel(item.category)}
                </span>
              </div>
            </figure>
          ))}
        </div>
      </div>
    </article>
  );
}
