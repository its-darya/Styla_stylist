/**
 * Isolated mock data layer.
 *
 * Every function here simulates a network round-trip and returns mock data.
 * Swap the bodies for real API calls later â€” the signatures and the shapes in
 * ./types.ts are the contract the UI depends on.
 */
import {
  type Category,
  type ReferenceMatchResult,
  type Outfit,
  type PinterestPin,
  type StyleId,
  type SuggestedProduct,
  type WardrobeItem,
} from "./types";

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

interface MatchResponse {
  matchedItems: {
    referenceImageUrl: string;
    wardrobeItem: {
      id: string;
      imageUrl: string;
      category: string;
      color: string;
      pattern: string;
      gender: string;
      dateAdded: string;
    };
    matchScore: number;
  }[];
  missingItems: {
    referenceImageUrl: string;
    category: string;
    suggestedProducts: SuggestedProduct[];
  }[];
}

export async function matchReferenceImage(
  file: File,
  _wardrobe: WardrobeItem[],
): Promise<ReferenceMatchResult> {
  const formData = new FormData();
  formData.append("file", file);
  return postReferenceMatch(formData);
}

export async function matchReferenceImageUrl(
  imageUrl: string,
  _wardrobe: WardrobeItem[],
): Promise<ReferenceMatchResult> {
  const formData = new FormData();
  formData.append("image_url", imageUrl);
  return postReferenceMatch(formData);
}

async function postReferenceMatch(formData: FormData): Promise<ReferenceMatchResult> {
  const response = await fetch("http://localhost:8000/api/reference/match", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Reference match failed: ${response.statusText}`);
  }

  const data: MatchResponse = await response.json();
  return {
    matchedItems: (data.matchedItems ?? []).map((m) => ({
      referenceImageUrl: m.referenceImageUrl,
      wardrobeItem: {
        id: m.wardrobeItem.id,
        imageUrl: m.wardrobeItem.imageUrl,
        category: m.wardrobeItem.category as Category,
        color: m.wardrobeItem.color || "Unknown",
        pattern: m.wardrobeItem.pattern || "Solid",
        gender: m.wardrobeItem.gender || "unisex",
        dateAdded: m.wardrobeItem.dateAdded || "",
      },
      matchScore: m.matchScore,
    })),
    missingItems: (data.missingItems ?? []).map((m) => ({
      referenceImageUrl: m.referenceImageUrl,
      category: (m.category || "top") as Category,
      suggestedProducts: m.suggestedProducts ?? [],
    })),
  };
}

export async function getPinterestFeed(): Promise<PinterestPin[]> {
  try {
    const response = await fetch("http://localhost:8000/api/pinterest/feed", {
      cache: "no-store",
    });
    if (!response.ok) return [];
    return (await response.json()) as PinterestPin[];
  } catch (error) {
    console.error("Failed to fetch Pinterest feed", error);
    return [];
  }
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

/**
 * Start a Kolors Virtual Try-On job.
 * Converts the person photo to base64 and sends it to the backend.
 * Returns { job_id, status, result_url } immediately (synchronous on the server side).
 */
export async function startTryOn(
  outfitId: string,
  items: WardrobeItem[],
  personFile: File,
): Promise<{ job_id: string; status: string; result_url?: string }> {
  // Convert file to base64
  const b64 = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(personFile);
  });

  const response = await fetch("http://localhost:8000/api/tryon", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      person_image_b64: b64,
      outfit_id: outfitId,
      items: items.map((item) => ({
        id: item.id,
        imageUrl: item.imageUrl,
        category: item.category,
        color: item.color,
      })),
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail ?? "Failed to start Try-On job");
  }
  return response.json();
}

export async function checkTryOn(
  jobId: string,
): Promise<{ status: string; result_url?: string; error?: string }> {
  const response = await fetch(`http://localhost:8000/api/tryon/${jobId}`);
  if (!response.ok) {
    throw new Error("Failed to check Try-On status");
  }
  return response.json();
}
