import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { toast } from "sonner";
import {
  deleteLookRemote,
  deleteWardrobeItem,
  getSavedLooks,
  getWardrobeItems,
  saveLookRemote,
} from "./api";
import { useAuth } from "./auth";
import type { Outfit, WardrobeItem } from "./types";

interface StylaState {
  wardrobe: WardrobeItem[];
  loadingWardrobe: boolean;
  addItem: (item: WardrobeItem) => void;
  removeItem: (id: string) => void;
  refreshWardrobe: () => Promise<void>;
  savedLooks: Outfit[];
  loadingLooks: boolean;
  saveLook: (outfit: Outfit) => Promise<void>;
  removeLook: (id: string) => Promise<void>;
}

const StylaContext = createContext<StylaState | null>(null);

export function StylaProvider({ children }: { children: ReactNode }) {
  const { status, user } = useAuth();
  const userId = status === "signed-in" ? user?.id ?? null : null;

  const [wardrobe, setWardrobe] = useState<WardrobeItem[]>([]);
  const [loadingWardrobe, setLoadingWardrobe] = useState(true);
  const [savedLooks, setSavedLooks] = useState<Outfit[]>([]);
  const [loadingLooks, setLoadingLooks] = useState(true);

  const refreshWardrobe = useCallback(async () => {
    if (!userId) return;
    setLoadingWardrobe(true);
    try {
      setWardrobe(await getWardrobeItems());
    } catch (error) {
      console.error("Failed to load wardrobe", error);
    } finally {
      setLoadingWardrobe(false);
    }
  }, [userId]);

  // (Re)load everything whenever the signed-in user changes.
  useEffect(() => {
    if (!userId) {
      setWardrobe([]);
      setSavedLooks([]);
      setLoadingWardrobe(status === "loading");
      setLoadingLooks(status === "loading");
      return;
    }
    let alive = true;
    setLoadingWardrobe(true);
    setLoadingLooks(true);
    getWardrobeItems()
      .then((items) => alive && setWardrobe(items))
      .catch((error) => console.error("Failed to load wardrobe", error))
      .finally(() => alive && setLoadingWardrobe(false));
    getSavedLooks()
      .then((looks) => alive && setSavedLooks(looks))
      .catch((error) => console.error("Failed to load saved looks", error))
      .finally(() => alive && setLoadingLooks(false));
    return () => {
      alive = false;
    };
  }, [userId, status]);

  const addItem = useCallback((item: WardrobeItem) => setWardrobe((w) => [item, ...w]), []);

  const removeItem = useCallback((id: string) => {
    setWardrobe((w) => w.filter((i) => i.id !== id));
    deleteWardrobeItem(id).catch((error) => {
      console.error(error);
      toast.error("Couldn't delete that item");
    });
  }, []);

  const saveLook = useCallback(async (outfit: Outfit) => {
    const saved = await saveLookRemote(outfit);
    setSavedLooks((l) => (l.some((o) => o.id === saved.id) ? l : [saved, ...l]));
  }, []);

  const removeLook = useCallback(async (id: string) => {
    setSavedLooks((l) => l.filter((o) => o.id !== id));
    try {
      await deleteLookRemote(id);
    } catch (error) {
      console.error(error);
      toast.error("Couldn't delete that look");
    }
  }, []);

  const value = useMemo(
    () => ({
      wardrobe,
      loadingWardrobe,
      addItem,
      removeItem,
      refreshWardrobe,
      savedLooks,
      loadingLooks,
      saveLook,
      removeLook,
    }),
    [wardrobe, loadingWardrobe, addItem, removeItem, refreshWardrobe, savedLooks, loadingLooks, saveLook, removeLook],
  );

  return <StylaContext.Provider value={value}>{children}</StylaContext.Provider>;
}

export function useStyla() {
  const ctx = useContext(StylaContext);
  if (!ctx) throw new Error("useStyla must be used inside StylaProvider");
  return ctx;
}
