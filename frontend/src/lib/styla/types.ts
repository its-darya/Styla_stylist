export type Category = "top" | "bottom" | "dress" | "outerwear" | "shoes" | "accessory";

export const CATEGORIES: { id: Category; label: string; plural: string }[] = [
  { id: "top", label: "Top", plural: "Tops" },
  { id: "bottom", label: "Bottom", plural: "Bottoms" },
  { id: "dress", label: "Dress", plural: "Dresses" },
  { id: "outerwear", label: "Outerwear", plural: "Outerwear" },
  { id: "shoes", label: "Shoes", plural: "Shoes" },
  { id: "accessory", label: "Accessory", plural: "Accessories" },
];

export interface User {
  id: string;
  email: string;
  name: string;
}

export interface WardrobeItem {
  id: string;
  imageUrl: string;
  /** Segmented garment crop when available, otherwise the same as imageUrl. */
  thumbnailUrl?: string;
  category: Category;
  /** Backend's fine-grained label, e.g. "jeans", "sweater". */
  fineCategory?: string;
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

export interface OutfitBreakdown {
  /** How well the outfit's palette works, on an absolute scale. */
  color: number;
  compatibility: number;
  style: number;
  personal: number | null;
  rules: number;
}

export interface Outfit {
  id: string;
  style: StyleId;
  items: WardrobeItem[];
  createdAt: string;
  /** Overall ranking score in [0, 1]. */
  score?: number;
  breakdown?: OutfitBreakdown;
  notes?: string[];
  /** One line naming the pieces, e.g. "navy shirt with grey jeans". */
  summary?: string;
  vtonImageUrl?: string;
}

/** How the generator should treat jackets and coats. */
export type OuterwearMode = "auto" | "always" | "never";

export interface GenerateOptions {
  gender?: string;
  usePersonalStyle?: boolean;
  count?: number;
  outerwear?: OuterwearMode;
  mustInclude?: string;
  excludeIds?: string[];
  offset?: number;
}

export interface SuggestedProduct {
  imageUrl: string;
  name: string;
  price: string;
  url: string;
  store?: string;
  /** Where a web result came from: google, duckduckgo or google-shopping. */
  source?: string;
}

export interface DetectedPiece {
  category: string;
  color: string;
  pattern: string;
  confidence: number;
}

export interface ReferencePiece extends DetectedPiece {
  slot: "top" | "bottom" | "dress";
  label: string;
  imageUrl: string;
  areaRatio: number;
}

export interface ReferenceMatch {
  slot?: string;
  referenceImageUrl: string;
  detected?: DetectedPiece;
  wardrobeItem: WardrobeItem;
  matchScore: number;
  alternates?: { wardrobeItem: WardrobeItem; matchScore: number }[];
}

export interface ReferenceMissing {
  slot?: string;
  referenceImageUrl: string;
  category: Category;
  detected?: DetectedPiece;
  closest?: { wardrobeItem: WardrobeItem; matchScore: number } | null;
  suggestedProducts: SuggestedProduct[];
  /** The web search used to find these products. */
  query?: string;
  googleShoppingUrl?: string;
}

export interface ReferenceMatchResult {
  sourceImageUrl?: string;
  pieces?: ReferencePiece[];
  matchedItems: ReferenceMatch[];
  missingItems: ReferenceMissing[];
  coverage?: number;
  // Web product search summary for the primary piece (ml/retrieval/web_shop.py).
  detected?: {
    category: string;
    color: string;
    pattern: string;
  };
  query?: string;
  googleShoppingUrl?: string;
  bestUrl?: string;
  onlineProducts?: SuggestedProduct[];
  referenceImageUrl?: string;
}

export interface PinterestPin {
  id: string;
  imageUrl: string;
  title?: string;
  link?: string;
}
