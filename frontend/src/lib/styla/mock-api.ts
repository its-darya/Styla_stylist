/**
 * Isolated mock data layer.
 *
 * Every function here simulates a network round-trip and returns mock data.
 * Swap the bodies for real API calls later â€” the signatures and the shapes in
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

const CATALOG: Omit<WardrobeItem, "id" | "dateAdded" | "gender">[] = [
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

];

const daysAgo = (n: number) => new Date(Date.now() - n * 86400000).toISOString();

export async function getWardrobeItems(): Promise<WardrobeItem[]> {
  try {
    const response = await fetch("http://localhost:8000/api/wardrobe", { cache: "no-store" });
    if (!response.ok) return [];
    
    const items = await response.json();
    return items.map((data: any) => {
      // Map specific backend categories to broad frontend categories
      const cat = (data.category || "top").toLowerCase();
      let mappedCategory: Category = "top";
      if (["pants", "jeans", "shorts", "skirt"].includes(cat)) mappedCategory = "bottom";
      else if (["dress"].includes(cat)) mappedCategory = "dress";
      else if (["jacket", "coat"].includes(cat)) mappedCategory = "outerwear";

      
      return {
        id: data.id,
        imageUrl: data.imageUrl,
        category: mappedCategory,
        color: data.color || "Unknown",
        pattern: data.pattern || "Solid",
        gender: data.gender || "unisex",
        dateAdded: data.dateAdded || new Date().toISOString(),
      };
    });
  } catch (error) {
    console.error("Failed to fetch wardrobe", error);
    return [];
  }
}

/** Connect to real FastAPI backend for garment analysis */
export async function analyzeGarmentPhoto(file: File): Promise<WardrobeItem> {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch("http://localhost:8000/api/wardrobe/upload", {
    method: "POST",
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }
  
  const data = await response.json();
  
  // Map specific backend categories to broad frontend categories
  const cat = data.category.toLowerCase();
  let mappedCategory: Category = "top";
  if (["pants", "jeans", "shorts", "skirt"].includes(cat)) mappedCategory = "bottom";
  else if (["dress"].includes(cat)) mappedCategory = "dress";
  else if (["jacket", "coat"].includes(cat)) mappedCategory = "outerwear";

  else if (["bag", "hat", "scarf", "sunglasses", "watch", "belt"].includes(cat)) mappedCategory = "top"; // Accessories fallback
  return {
    id: data.id,
    imageUrl: `http://localhost:8000${data.filename}`,
    category: mappedCategory,
    color: data.color,
    pattern: data.pattern || "Solid",
    gender: data.gender || "unisex",
    dateAdded: new Date().toISOString(),
  };
}

export async function deleteWardrobeItem(id: string): Promise<void> {
  const response = await fetch(`http://localhost:8000/api/wardrobe/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Delete failed: ${response.statusText}`);
  }
}

const sample = <T,>(arr: T[], n: number) =>
  [...arr].sort(() => Math.random() - 0.5).slice(0, n);

export async function generateOutfit(style: StyleId, wardrobe: WardrobeItem[], userId?: string, gender: string = "any"): Promise<Outfit[]> {
  const payload: any = { style, gender };
  if (userId) {
    payload.user_id = userId;
  }
  
  try {
    const response = await fetch("http://localhost:8000/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    
    if (!response.ok) {
      throw new Error(`Generate failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    if (!data || data.length === 0) {
      return [];
    }
    
    // Return all results mapped to frontend types
    return data.map((result: any, i: number) => {
      const mappedItems = result.items.map((item: any) => {
        let mappedCategory: Category = "top";
        const cat = (item.category || "top").toLowerCase();
        if (["pants", "jeans", "shorts", "skirt"].includes(cat)) mappedCategory = "bottom";
        else if (["dress"].includes(cat)) mappedCategory = "dress";
        else if (["jacket", "coat"].includes(cat)) mappedCategory = "outerwear";
  
        
        return {
          id: item.id,
          imageUrl: item.imageUrl,
          category: mappedCategory,
          color: item.color || "Unknown",
          pattern: item.pattern || "Solid",
          dateAdded: new Date().toISOString(),
        };
      });

      return {
        id: result.id || `of_${Date.now().toString(36)}_${i}`,
        style,
        items: mappedItems,
        createdAt: new Date().toISOString(),
      };
    });
  } catch (err) {
    console.error("Failed to generate outfit", err);
    return [];
  }
}

export async function matchReferenceImage(
  _image: File,
  wardrobe: WardrobeItem[],
): Promise<ReferenceMatchResult> {
  await delay(2800);
  const matchedSource = sample(wardrobe, Math.min(2, wardrobe.length));
  const usedCategories = new Set(matchedSource.map((i) => i.category));
  const missingCategory =
    CATEGORIES.find((c) => !usedCategories.has(c.id))?.id ?? ("top" as Category);

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

export async function uploadPersonalStyleRef(file: File, userId: string = "default_user"): Promise<boolean> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("user_id", userId);
  
  try {
    const response = await fetch("http://localhost:8000/api/style/personal/upload", {
      method: "POST",
      body: formData,
    });
    return response.ok;
  } catch (err) {
    console.error("Failed to upload personal style ref", err);
    return false;
  }
}

export async function startTryOn(outfitId: string, avatarId: string, items: WardrobeItem[]): Promise<string> {
  const response = await fetch("http://localhost:8000/api/tryOn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      outfit_id: outfitId,
      avatar_id: avatarId,
      items: items
    })
  });
  if (!response.ok) {
    throw new Error("Failed to start Try-On job");
  }
  const data = await response.json();
  return data.job_id;
}

export async function checkTryOn(jobId: string): Promise<{status: string, result_url?: string, error?: string}> {
  const response = await fetch(`http://localhost:8000/api/tryOn/${jobId}`);
  if (!response.ok) {
    throw new Error("Failed to check Try-On status");
  }
  return response.json();
}
