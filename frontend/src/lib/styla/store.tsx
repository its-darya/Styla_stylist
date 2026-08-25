import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getWardrobeItems, deleteWardrobeItem } from "./mock-api";
import type { Outfit, WardrobeItem } from "./types";

interface StylaState {
  wardrobe: WardrobeItem[];
  loadingWardrobe: boolean;
  addItem: (item: WardrobeItem) => void;
  removeItem: (id: string) => void;
  savedLooks: Outfit[];
  saveLook: (outfit: Outfit) => void;
  removeLook: (id: string) => void;
}

const StylaContext = createContext<StylaState | null>(null);

export function StylaProvider({ children }: { children: ReactNode }) {
  const [wardrobe, setWardrobe] = useState<WardrobeItem[]>([]);
  const [loadingWardrobe, setLoading] = useState(true);
  const [savedLooks, setSavedLooks] = useState<Outfit[]>([]);

  useEffect(() => {
    let alive = true;
    getWardrobeItems().then((items) => {
      if (!alive) return;
      setWardrobe(items);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  const addItem = useCallback((item: WardrobeItem) => setWardrobe((w) => [item, ...w]), []);
  const removeItem = useCallback(
    (id: string) => {
      setWardrobe((w) => w.filter((i) => i.id !== id));
      deleteWardrobeItem(id).catch(console.error);
    },
    [],
  );
  const saveLook = useCallback(
    (outfit: Outfit) =>
      setSavedLooks((l) => (l.some((o) => o.id === outfit.id) ? l : [outfit, ...l])),
    [],
  );
  const removeLook = useCallback((id: string) => setSavedLooks((l) => l.filter((o) => o.id !== id)), []);

  const value = useMemo(
    () => ({ wardrobe, loadingWardrobe, addItem, removeItem, savedLooks, saveLook, removeLook }),
    [wardrobe, loadingWardrobe, addItem, removeItem, savedLooks, saveLook, removeLook],
  );

  return <StylaContext.Provider value={value}>{children}</StylaContext.Provider>;
}

export function useStyla() {
  const ctx = useContext(StylaContext);
  if (!ctx) throw new Error("useStyla must be used inside StylaProvider");
  return ctx;
}
