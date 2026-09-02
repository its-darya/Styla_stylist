import { Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { CATEGORIES, type WardrobeItem } from "@/lib/styla/types";

export function categoryLabel(id: WardrobeItem["category"]) {
  return CATEGORIES.find((c) => c.id === id)?.label ?? id;
}

export function ItemCard({
  item,
  onDelete,
}: {
  item: WardrobeItem;
  onDelete?: (id: string) => void;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button className="glass group overflow-hidden rounded-3xl p-0 text-left transition-transform hover:-translate-y-1">
          <div className="aspect-[3/4] overflow-hidden bg-muted">
            <img
              src={item.imageUrl}
              alt={`${item.color} ${categoryLabel(item.category)}`}
              loading="lazy"
              className="size-full object-cover transition-transform duration-500 group-hover:scale-105"
            />
          </div>
          <div className="flex items-center justify-between gap-2 p-3">
            <span className="truncate text-sm font-medium">{categoryLabel(item.category)}</span>
            <span className="shrink-0 rounded-full bg-accent-soft px-2 py-0.5 text-[11px] text-primary">
              {item.color}
            </span>
          </div>
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-md rounded-3xl">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">
            {item.color} {categoryLabel(item.category)}
          </DialogTitle>
        </DialogHeader>
        <img
          src={item.imageUrl}
          alt={`${item.color} ${categoryLabel(item.category)}`}
          className="aspect-[3/4] w-full rounded-2xl object-cover"
        />
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <Detail label="Category" value={categoryLabel(item.category)} />
          <Detail label="Colour" value={item.color} />
          <Detail label="Pattern" value={item.pattern} />
          <Detail label="Gender" value={item.gender ? item.gender.charAt(0).toUpperCase() + item.gender.slice(1) : "Unisex"} />
          <Detail
            label="Added"
            value={new Date(item.dateAdded).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
            })}
          />
        </dl>
        {onDelete && (
          <Button variant="outline" onClick={() => onDelete(item.id)} className="rounded-full">
            <Trash2 className="size-4" /> Remove from wardrobe
          </Button>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-secondary/70 px-3 py-2">
      <dt className="text-[11px] uppercase tracking-widest text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 font-medium">{value}</dd>
    </div>
  );
}
