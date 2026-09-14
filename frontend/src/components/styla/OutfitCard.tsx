import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { Outfit } from "@/lib/styla/types";
import { STYLES } from "@/lib/styla/types";

function label(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/** Head to toe, so a look reads the way it is worn. */
const STACK_ORDER: Record<string, number> = {
  hat: 1, sunglasses: 2, scarf: 3, coat: 4, outerwear: 4, jacket: 5,
  sweater: 6, shirt: 7, top: 7, "t-shirt": 8, dress: 9,
  belt: 10, pants: 11, jeans: 12, bottom: 12, skirt: 13, shorts: 14, bag: 15,
};

function orderOf(item: { category: string; fineCategory?: string }) {
  return STACK_ORDER[item.fineCategory ?? ""] ?? STACK_ORDER[item.category] ?? 99;
}

/** Reading order for the sentence: the base piece, then the bottom, then the
    layer that goes over them — which is not the head-to-toe display order. */
const DESCRIBE_ORDER: Record<string, number> = {
  top: 0, dress: 0, bottom: 1, shoes: 2, accessory: 3, outerwear: 4,
};

/** Saved looks are stored without the generator's sentence, so rebuild it. */
function describe(items: Outfit["items"]) {
  const names = [...items]
    .sort((a, b) => (DESCRIBE_ORDER[a.category] ?? 9) - (DESCRIBE_ORDER[b.category] ?? 9))
    .map((i) => `${i.color} ${i.fineCategory || i.category}`.toLowerCase());
  if (names.length === 0) return "";
  if (names.length === 1) return names[0]!;
  if (names.length === 2) return `${names[0]} with ${names[1]}`;
  return `${names[0]} with ${names[1]}, layered under ${names[names.length - 1]}`;
}

function Bar({ name, value }: { name: string; value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 text-[11px] uppercase tracking-wider text-muted-foreground">
        {name}
      </span>
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
        <span
          className="block h-full rounded-full bg-primary/70"
          style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }}
        />
      </span>
    </div>
  );
}

export function OutfitCard({
  outfit,
  action,
  footer,
  selected,
  onSelect,
  showBreakdown = false,
  compact = false,
}: {
  outfit: Outfit;
  action?: ReactNode;
  footer?: ReactNode;
  selected?: boolean;
  onSelect?: () => void;
  showBreakdown?: boolean;
  compact?: boolean;
}) {
  const style = STYLES.find((s) => s.id === outfit.style);
  const percent = typeof outfit.score === "number" ? Math.round(outfit.score * 100) : null;
  const items = [...outfit.items].sort((a, b) => orderOf(a) - orderOf(b));

  return (
    <article
      className={cn(
        "glass overflow-hidden rounded-3xl p-4 transition-all",
        onSelect && "cursor-pointer hover:-translate-y-0.5 hover:shadow-lg",
        selected && "ring-2 ring-primary",
      )}
      onClick={onSelect}
    >
      <div className="mb-3 grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-display text-xl">{style?.label ?? outfit.style}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {outfit.summary || describe(items)}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {percent !== null && (
            <span className="rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-primary">
              {percent}%
            </span>
          )}
          {action}
        </div>
      </div>

      <div
        className={cn(
          "flex justify-center rounded-2xl bg-white/60 p-4",
          compact ? "min-h-[180px]" : "min-h-[260px]",
        )}
      >
        <div className="flex w-full flex-wrap items-center justify-center gap-3">
          {items.map((item) => {
            const name = label(item.fineCategory || item.category);
            return (
              <figure
                key={item.id}
                className={cn("group/item relative", compact ? "w-20" : "w-28 md:w-32")}
                title={`${item.color} ${name}`}
              >
                <img
                  src={item.thumbnailUrl ?? item.imageUrl}
                  alt={`${item.color} ${name}`}
                  loading="lazy"
                  className={cn(
                    "w-full object-contain drop-shadow-sm",
                    compact ? "h-24" : "h-36",
                  )}
                />
                <figcaption className="mt-1 truncate text-center text-[10px] uppercase tracking-wider text-muted-foreground">
                  {item.color} {name}
                </figcaption>
              </figure>
            );
          })}
        </div>
      </div>

      {outfit.notes && outfit.notes.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {outfit.notes.map((note) => (
            <span
              key={note}
              className="rounded-full bg-secondary/70 px-2.5 py-0.5 text-[11px] text-muted-foreground"
            >
              {note}
            </span>
          ))}
        </div>
      )}

      {footer && <div className="mt-3 flex flex-wrap gap-2">{footer}</div>}

      {showBreakdown && outfit.breakdown && (
        <div className="mt-3 space-y-1.5 border-t border-border/50 pt-3">
          <Bar name="Colour" value={outfit.breakdown.color ?? outfit.breakdown.compatibility} />
          <Bar name="Style fit" value={outfit.breakdown.style} />
          <Bar name="Pairing" value={outfit.breakdown.compatibility} />
          {outfit.breakdown.personal !== null && (
            <Bar name="Your taste" value={outfit.breakdown.personal} />
          )}
        </div>
      )}
    </article>
  );
}
