/**
 * Styla API client.
 *
 * Every call goes to the FastAPI backend (`VITE_API_URL`, default
 * http://localhost:8000). Authenticated calls attach the bearer token that
 * `auth.tsx` registers via `configureApiAuth`; a 401 signs the user out.
 */
import {
  type Category,
  type GenerateOptions,
  type Outfit,
  type PinterestPin,
  type ReferenceMatchResult,
  type StyleId,
  type User,
  type WardrobeItem,
} from "./types";

const envUrl = (import.meta.env["VITE_API_URL"] as string | undefined)?.trim();
export const API_BASE = (envUrl && envUrl.length > 0 ? envUrl : "http://localhost:8000").replace(
  /\/$/,
  "",
);

let getToken: () => string | null = () => null;
let handleUnauthorized: () => void = () => {};

export function configureApiAuth(tokenGetter: () => string | null, onUnauthorized: () => void) {
  getToken = tokenGetter;
  handleUnauthorized = onUnauthorized;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

interface RequestOptions extends RequestInit {
  auth?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { auth = true, headers, ...init } = options;
  const finalHeaders = new Headers(headers ?? {});
  if (auth) {
    const token = getToken();
    if (token) finalHeaders.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: finalHeaders });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        const first = body.detail[0] as { msg?: string } | undefined;
        if (first?.msg) detail = first.msg;
      }
    } catch {
      /* non-JSON error body */
    }
    if (response.status === 401 && auth) handleUnauthorized();
    throw new ApiError(response.status, detail || `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function jsonBody(body: unknown): RequestOptions {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

export function mapCategory(raw: string | undefined | null): Category {
  const cat = (raw ?? "top").toLowerCase();
  if (["bottom", "pants", "jeans", "shorts", "skirt"].includes(cat)) return "bottom";
  if (cat === "dress") return "dress";
  if (["outerwear", "jacket", "coat"].includes(cat)) return "outerwear";
  if (["shoes", "sneakers", "boots", "heels", "sandals"].includes(cat)) return "shoes";
  if (["accessory", "bag", "hat", "scarf", "sunglasses", "watch", "belt"].includes(cat)) return "accessory";
  return "top";
}

interface RawItem {
  id: string;
  imageUrl?: string;
  thumbnailUrl?: string;
  filename?: string;
  category?: string;
  fineCategory?: string;
  color?: string;
  pattern?: string;
  gender?: string;
  dateAdded?: string | null;
}

export function mapItem(data: RawItem): WardrobeItem {
  const imageUrl = data.imageUrl ?? (data.filename ? `${API_BASE}${data.filename}` : "");
  return {
    id: data.id,
    imageUrl,
    thumbnailUrl: data.thumbnailUrl ?? imageUrl,
    category: mapCategory(data.category),
    fineCategory: data.fineCategory ?? data.category ?? "",
    color: data.color || "Unknown",
    pattern: data.pattern || "Solid",
    gender: data.gender || "unisex",
    dateAdded: data.dateAdded || new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export interface AuthPayload {
  token: string;
  user: User;
}

export function signup(email: string, password: string, name: string): Promise<AuthPayload> {
  return request<AuthPayload>("/api/auth/signup", { ...jsonBody({ email, password, name }), auth: false });
}

export function login(email: string, password: string): Promise<AuthPayload> {
  return request<AuthPayload>("/api/auth/login", { ...jsonBody({ email, password }), auth: false });
}

export function fetchMe(): Promise<User> {
  return request<User>("/api/auth/me");
}

// ---------------------------------------------------------------------------
// Wardrobe
// ---------------------------------------------------------------------------

export async function getWardrobeItems(): Promise<WardrobeItem[]> {
  const items = await request<RawItem[]>("/api/wardrobe", { cache: "no-store" });
  return items.map(mapItem);
}

export async function analyzeGarmentPhoto(file: File): Promise<WardrobeItem> {
  const formData = new FormData();
  formData.append("file", file);
  const data = await request<RawItem>("/api/wardrobe/upload", { method: "POST", body: formData });
  return mapItem(data);
}

export async function deleteWardrobeItem(id: string): Promise<void> {
  await request(`/api/wardrobe/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Outfit generation
// ---------------------------------------------------------------------------

interface RawOutfit {
  id: string;
  style: string;
  items: RawItem[];
  createdAt: string;
  score?: number;
  breakdown?: Outfit["breakdown"];
  notes?: string[];
  summary?: string;
}

export async function generateOutfit(
  style: StyleId,
  options: GenerateOptions = {},
): Promise<Outfit[]> {
  const data = await request<RawOutfit[]>(
    "/api/generate",
    jsonBody({
      style,
      gender: options.gender ?? "any",
      use_personal_style: options.usePersonalStyle ?? false,
      count: options.count ?? 8,
      outerwear: options.outerwear ?? "auto",
      must_include: options.mustInclude ?? null,
      exclude_ids: options.excludeIds ?? [],
      offset: options.offset ?? 0,
    }),
  );
  return (data ?? []).map((o, i) => {
    const outfit: Outfit = {
      id: o.id || `of_${Date.now().toString(36)}_${i}`,
      style,
      items: o.items.map(mapItem),
      createdAt: o.createdAt || new Date().toISOString(),
    };
    if (typeof o.score === "number") outfit.score = o.score;
    if (o.breakdown) outfit.breakdown = o.breakdown;
    if (o.notes) outfit.notes = o.notes;
    if (o.summary) outfit.summary = o.summary;
    return outfit;
  });
}

// ---------------------------------------------------------------------------
// Personal style references
// ---------------------------------------------------------------------------

export async function uploadPersonalStyleRef(file: File): Promise<number> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await request<{ success: boolean; count: number }>("/api/style/personal/upload", {
    method: "POST",
    body: formData,
  });
  return res.count;
}

export async function getPersonalStyleCount(): Promise<number> {
  const res = await request<{ count: number }>("/api/style/personal");
  return res.count;
}

export async function clearPersonalStyle(): Promise<void> {
  await request("/api/style/personal", { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Reference matching
// ---------------------------------------------------------------------------

export function matchReferenceImage(file: File): Promise<ReferenceMatchResult> {
  const formData = new FormData();
  formData.append("file", file);
  return postReferenceMatch(formData);
}

export function matchReferenceImageUrl(imageUrl: string): Promise<ReferenceMatchResult> {
  const formData = new FormData();
  formData.append("image_url", imageUrl);
  return postReferenceMatch(formData);
}

interface RawMatchResponse {
  sourceImageUrl?: string;
  pieces?: ReferenceMatchResult["pieces"];
  coverage?: number;
  matchedItems: {
    slot?: string;
    referenceImageUrl: string;
    detected?: ReferenceMatchResult["matchedItems"][number]["detected"];
    wardrobeItem: RawItem;
    matchScore: number;
    alternates?: { wardrobeItem: RawItem; matchScore: number }[];
  }[];
  missingItems: {
    slot?: string;
    referenceImageUrl: string;
    category: string;
    detected?: ReferenceMatchResult["missingItems"][number]["detected"];
    closest?: { wardrobeItem: RawItem; matchScore: number } | null;
    suggestedProducts: ReferenceMatchResult["missingItems"][number]["suggestedProducts"];
  }[];
}

async function postReferenceMatch(formData: FormData): Promise<ReferenceMatchResult> {
  const data = await request<RawMatchResponse>("/api/reference/match", {
    method: "POST",
    body: formData,
  });
  const result: ReferenceMatchResult = {
    matchedItems: (data.matchedItems ?? []).map((m) => {
      const entry: ReferenceMatchResult["matchedItems"][number] = {
        referenceImageUrl: m.referenceImageUrl,
        wardrobeItem: mapItem(m.wardrobeItem),
        matchScore: m.matchScore,
        alternates: (m.alternates ?? []).map((a) => ({
          wardrobeItem: mapItem(a.wardrobeItem),
          matchScore: a.matchScore,
        })),
      };
      if (m.slot) entry.slot = m.slot;
      if (m.detected) entry.detected = m.detected;
      return entry;
    }),
    missingItems: (data.missingItems ?? []).map((m) => {
      const entry: ReferenceMatchResult["missingItems"][number] = {
        referenceImageUrl: m.referenceImageUrl,
        category: mapCategory(m.category),
        closest: m.closest
          ? { wardrobeItem: mapItem(m.closest.wardrobeItem), matchScore: m.closest.matchScore }
          : null,
        suggestedProducts: m.suggestedProducts ?? [],
      };
      if (m.slot) entry.slot = m.slot;
      if (m.detected) entry.detected = m.detected;
      return entry;
    }),
  };
  if (data.sourceImageUrl) result.sourceImageUrl = data.sourceImageUrl;
  if (data.pieces) result.pieces = data.pieces;
  if (typeof data.coverage === "number") result.coverage = data.coverage;
  return result;
}

// ---------------------------------------------------------------------------
// Inspiration feed (public)
// ---------------------------------------------------------------------------

export async function getPinterestFeed(): Promise<PinterestPin[]> {
  try {
    return await request<PinterestPin[]>("/api/pinterest/feed", { cache: "no-store", auth: false });
  } catch (error) {
    console.error("Failed to fetch inspiration feed", error);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Saved looks
// ---------------------------------------------------------------------------

interface RawLook {
  id: string;
  style: string;
  items: RawItem[];
  score?: number | null;
  createdAt?: string | null;
}

function mapLook(raw: RawLook): Outfit {
  const outfit: Outfit = {
    id: raw.id,
    style: raw.style as StyleId,
    items: raw.items.map(mapItem),
    createdAt: raw.createdAt ?? new Date().toISOString(),
  };
  if (typeof raw.score === "number") outfit.score = raw.score;
  return outfit;
}

export async function getSavedLooks(): Promise<Outfit[]> {
  const rows = await request<RawLook[]>("/api/looks", { cache: "no-store" });
  return rows.map(mapLook);
}

export async function saveLookRemote(outfit: Outfit): Promise<Outfit> {
  const row = await request<RawLook>(
    "/api/looks",
    jsonBody({ id: outfit.id, style: outfit.style, items: outfit.items, score: outfit.score ?? null }),
  );
  return mapLook(row);
}

export async function deleteLookRemote(id: string): Promise<void> {
  await request(`/api/looks/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Virtual try-on
// ---------------------------------------------------------------------------

export async function fileToBase64(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const payload = result.split(",")[1];
      if (payload === undefined) reject(new Error("Could not read file"));
      else resolve(payload);
    };
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

export interface TryOnResult {
  job_id?: string;
  status: string;
  result_url?: string;
  /** Pieces the service actually rendered onto the photo. */
  applied?: string[];
  /** Pieces it could not render this time, so they are missing from the image. */
  skipped?: string[];
  /** "ai" when a hosted model produced the image, "preview" when it was drawn
      locally because those models were unavailable. */
  mode?: "ai" | "preview";
  error?: string;
}

export async function startTryOn(
  outfitId: string,
  items: WardrobeItem[],
  personFile: File | null,
): Promise<TryOnResult> {
  const person_image_b64 = personFile ? await fileToBase64(personFile) : "";
  return request(
    "/api/tryon",
    jsonBody({
      person_image_b64,
      outfit_id: outfitId,
      items: items.map((item) => ({
        id: item.id,
        imageUrl: item.imageUrl,
        category: item.category,
        color: item.color,
        gender: item.gender,
      })),
    }),
  );
}

export function checkTryOn(jobId: string): Promise<TryOnResult> {
  return request(`/api/tryon/${jobId}`);
}
