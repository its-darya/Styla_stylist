import { useMemo, useState } from "react";
import { Check, Search, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { CATEGORIES, type Category, type WardrobeItem } from "@/lib/styla/types";

function itemName(item: WardrobeItem) {
  const fine = item.fineCategory?.trim();
  const type = fine ? fine.charAt(0).toUpperCase() + fine.slice(1) : item.category;
  return `${item.color} ${type}`;
}

/**
 * Pick one garment to build outfits around. Opens the wardrobe in a dialog
 * rather than a dropdown, because a hundred items are far easier to
 * recognise by their photo than by a line of text.
 */
export function ItemPicker({
  wardrobe,
  selectedId,
  onSelect,
  trigger,
}: {
  wardrobe: WardrobeItem[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  trigger: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Category | "all">("all");

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return wardrobe.filter((i) => {
      if (filter !== "all" && i.category !== filter) return false;
      if (!q) return true;
      return itemName(i).toLowerCase().includes(q) || i.pattern.toLowerCase().includes(q);
    });
  }, [wardrobe, query, filter]);

  const chips: { id: Category | "all"; label: string }[] = [
    { id: "all", label: "All" },
    ...CATEGORIES.filter((c) => wardrobe.some((i) => i.category === c.id)).map((c) => ({
      id: c.id as Category | "all",
      label: c.plural,
    })),
  ];

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-hidden rounded-3xl">
        <DialogHeader>
          <DialogTitle className="font-display text-2xl">Build outfits around…</DialogTitle>
        </DialogHeader>

        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by colour or type"
            className="h-10 rounded-xl pl-9"
          />
        </div>

        <div className="flex flex-wrap gap-1.5">
          {chips.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => setFilter(c.id)}
              className={cn(
                "rounded-full border border-border/70 px-3 py-1 text-xs transition-colors",
                filter === c.id
                  ? "border-primary bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {c.label}
            </button>
          ))}
          {selectedId && (
            <button
              type="button"
              onClick={() => {
                onSelect(null);
                setOpen(false);
              }}
              className="ml-auto flex items-center gap-1 rounded-full px-3 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              <X className="size-3" /> Clear
            </button>
          )}
        </div>

        {/* A definite height with content-sized rows: inside the dialog's own
            capped grid, `max-h` alone lets the rows collapse to a few pixels. */}
        <div className="grid h-[55vh] min-h-0 auto-rows-max grid-cols-3 gap-3 overflow-y-auto pr-1 sm:grid-cols-4 md:grid-cols-5">
          {visible.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelect(item.id === selectedId ? null : item.id);
                setOpen(false);
              }}
              className={cn(
                "group relative overflow-hidden rounded-2xl border bg-white p-1 text-left transition-all hover:-translate-y-0.5 hover:shadow-md",
                item.id === selectedId ? "border-primary ring-2 ring-primary" : "border-border/60",
              )}
            >
              <img
                src={item.thumbnailUrl ?? item.imageUrl}
                alt={itemName(item)}
                loading="lazy"
                className="aspect-[3/4] w-full object-contain"
              />
              <span className="block truncate px-1 pb-1 text-[11px] text-muted-foreground">
                {itemName(item)}
              </span>
              {item.id === selectedId && (
                <span className="absolute right-2 top-2 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                  <Check className="size-3" />
                </span>
              )}
            </button>
          ))}
          {visible.length === 0 && (
            <p className="col-span-full py-10 text-center text-sm text-muted-foreground">
              Nothing matches that search.
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
