export type Category = "top" | "bottom" | "dress" | "outerwear";

export const CATEGORIES: { id: Category; label: string; plural: string }[] = [
  { id: "top", label: "Top", plural: "Tops" },
  { id: "bottom", label: "Bottom", plural: "Bottoms" },
  { id: "dress", label: "Dress", plural: "Dresses" },
  { id: "outerwear", label: "Outerwear", plural: "Outerwear" },
];

export interface WardrobeItem {
  id: string;
  imageUrl: string;
  category: Category;
  color: string;
  pattern: string;
  gender: string;
  dateAdded: string;
}

export type StyleId =
  | "casual"
  | "formal"
  | "business-casual"
  | "streetwear"
  | "sporty"
  | "bohemian"
  | "minimalist"
  | "elegant";

export interface StyleOption {
  id: StyleId;
  label: string;
  hint: string;
  tint: string;
}

export const STYLES: StyleOption[] = [
  { id: "casual", label: "Casual", hint: "Easy everyday", tint: "oklch(0.82 0.09 60)" },
  { id: "formal", label: "Formal", hint: "Sharp tailoring", tint: "oklch(0.55 0.06 265)" },
  { id: "business-casual", label: "Business Casual", hint: "Smart but soft", tint: "oklch(0.72 0.07 200)" },
  { id: "streetwear", label: "Streetwear", hint: "Loud layers", tint: "oklch(0.68 0.17 30)" },
  { id: "sporty", label: "Sporty", hint: "Move-ready", tint: "oklch(0.78 0.15 145)" },
  { id: "bohemian", label: "Bohemian", hint: "Free & flowy", tint: "oklch(0.75 0.12 90)" },
  { id: "minimalist", label: "Minimalist", hint: "Quiet lines", tint: "oklch(0.85 0.02 250)" },
  { id: "elegant", label: "Elegant / Evening", hint: "After dark", tint: "oklch(0.5 0.12 320)" },
];

export interface Outfit {
  id: string;
  style: StyleId;
  items: WardrobeItem[];
  createdAt: string;
  vtonImageUrl?: string;
}

export interface SuggestedProduct {
  imageUrl: string;
  name: string;
  price: string;
  url: string;
}

export interface ReferenceMatchResult {
  matchedItems: {
    referenceImageUrl: string;
    wardrobeItem: WardrobeItem;
    matchScore: number;
  }[];
  missingItems: {
    referenceImageUrl: string;
    category: Category;
    suggestedProducts: SuggestedProduct[];
  }[];
}
