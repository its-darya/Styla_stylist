/**
 * Isolated mock data layer.
 *
 * Every function here simulates a network round-trip and returns mock data.
 * Swap the bodies for real API calls later — the signatures and the shapes in
 * ./types.ts are the contract the UI depends on.
 */
import {
  CATEGORIES,
  type Category,
  type ReferenceMatchResult,
  type Outfit,
  type StyleId,
  type WardrobeItem,
} from "./types";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

const img = (id: string, w = 800) =>
  `https://images.unsplash.com/photo-${id}?auto=format&fit=crop&w=${w}&q=70`;

const COLORS = ["Ecru", "Charcoal", "Sand", "Off-white", "Slate", "Espresso", "Olive", "Rust"];
const PATTERNS = ["Solid", "Ribbed", "Pinstripe", "Check", "Textured"];

let seed = 1;
const uid = () => `wi_${Date.now().toString(36)}_${(seed++).toString(36)}`;

const CATALOG: Omit<WardrobeItem, "id" | "dateAdded">[] = [
  { imageUrl: img("1521572163474-6864f9cf17ab"), category: "top", color: "Off-white", pattern: "Solid" },
  { imageUrl: img("1596755094514-f87e34085b2c"), category: "top", color: "Sand", pattern: "Ribbed" },
  { imageUrl: img("1620799140408-edc6dcb6d633"), category: "top", color: "Charcoal", pattern: "Solid" },
  { imageUrl: img("1541099649105-f69ad21f3246"), category: "top", color: "Ecru", pattern: "Textured" },
  { imageUrl: img("1542272604-787c3835535d"), category: "bottom", color: "Indigo", pattern: "Solid" },
  { imageUrl: img("1594633312681-425c7b97ccd1"), category: "bottom", color: "Espresso", pattern: "Check" },
  { imageUrl: img("1624378439575-d8705ad7ae80"), category: "bottom", color: "Slate", pattern: "Pinstripe" },
  { imageUrl: img("1595777457583-95e059d581b8"), category: "dress", color: "Rust", pattern: "Solid" },
  { imageUrl: img("1572804013309-59a88b7e92f1"), category: "dress", color: "Charcoal", pattern: "Solid" },
  { imageUrl: img("1591047139829-d91aecb6caea"), category: "outerwear", color: "Olive", pattern: "Solid" },
  { imageUrl: img("1544022613-e87ca75a784a"), category: "outerwear", color: "Sand", pattern: "Textured" },
  { imageUrl: img("1549298916-b41d501d3772"), category: "shoes", color: "Off-white", pattern: "Solid" },
  { imageUrl: img("1560769629-975ec94e6a86"), category: "shoes", color: "Espresso", pattern: "Solid" },
];

const daysAgo = (n: number) => new Date(Date.now() - n * 86400000).toISOString();

export async function getWardrobeItems(): Promise<WardrobeItem[]> {
  await delay(700);
  return CATALOG.map((c, i) => ({
    ...c,
    id: `seed_${i}`,
    dateAdded: daysAgo(i + 1),
  }));
}

/** Mock "segmentation + classification" of an uploaded garment photo. */
export async function analyzeGarmentPhoto(file: File): Promise<WardrobeItem> {
  await delay(2400);
  const pick = CATALOG[Math.floor(Math.random() * CATALOG.length)]!;
  return {
    id: uid(),
    imageUrl: URL.createObjectURL(file),
    category: pick.category,
    color: COLORS[Math.floor(Math.random() * COLORS.length)]!,
    pattern: PATTERNS[Math.floor(Math.random() * PATTERNS.length)]!,
    dateAdded: new Date().toISOString(),
  };
}

const sample = <T,>(arr: T[], n: number) =>
  [...arr].sort(() => Math.random() - 0.5).slice(0, n);

export async function generateOutfit(style: StyleId, wardrobe: WardrobeItem[]): Promise<Outfit> {
  await delay(1900);
  const dress = wardrobe.filter((i) => i.category === "dress");
  const tops = wardrobe.filter((i) => i.category === "top");
  const bottoms = wardrobe.filter((i) => i.category === "bottom");
  const outer = wardrobe.filter((i) => i.category === "outerwear");
  const shoes = wardrobe.filter((i) => i.category === "shoes");

  const items: WardrobeItem[] = [];
  const useDress = dress.length > 0 && Math.random() > 0.6;
  if (useDress) items.push(...sample(dress, 1));
  else items.push(...sample(tops, 1), ...sample(bottoms, 1));
  if (shoes.length && Math.random() > 0.25) items.push(...sample(shoes, 1));
  if (outer.length && items.length < 4 && Math.random() > 0.4) items.push(...sample(outer, 1));

  return {
    id: `of_${Date.now().toString(36)}`,
    style,
    items: items.length ? items : sample(wardrobe, Math.min(2, wardrobe.length)),
    createdAt: new Date().toISOString(),
  };
}

export async function matchReferenceImage(
  _image: File,
  wardrobe: WardrobeItem[],
): Promise<ReferenceMatchResult> {
  await delay(2800);
  const matchedSource = sample(wardrobe, Math.min(2, wardrobe.length));
  const usedCategories = new Set(matchedSource.map((i) => i.category));
  const missingCategory =
    CATEGORIES.find((c) => !usedCategories.has(c.id))?.id ?? ("shoes" as Category);

  return {
    matchedItems: matchedSource.map((wardrobeItem, i) => ({
      referenceImageUrl: CATALOG[(i * 5 + 2) % CATALOG.length]!.imageUrl,
      wardrobeItem,
      matchScore: 72 + Math.floor(Math.random() * 26),
    })),
    missingItems: [
      {
        referenceImageUrl: img("1543163521-1bf539c55dd2"),
        category: missingCategory,
        suggestedProducts: [
          { imageUrl: img("1560769629-975ec94e6a86", 400), name: "Leather Loafer", price: "$128", url: "#" },
          { imageUrl: img("1549298916-b41d501d3772", 400), name: "Canvas Low Sneaker", price: "$74", url: "#" },
          { imageUrl: img("1600185365483-26d7a4cc7519", 400), name: "Runner, Bone", price: "$96", url: "#" },
        ],
      },
      {
        referenceImageUrl: img("1591047139829-d91aecb6caea"),
        category: "outerwear",
        suggestedProducts: [
          { imageUrl: img("1544022613-e87ca75a784a", 400), name: "Oversized Wool Coat", price: "$210", url: "#" },
          { imageUrl: img("1520975954732-35dd22299614", 400), name: "Cropped Utility Jacket", price: "$139", url: "#" },
        ],
      },
    ],
  };
}
