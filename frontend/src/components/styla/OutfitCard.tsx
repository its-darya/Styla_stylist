import { useState, useEffect } from "react";
import { STYLES, type Outfit } from "@/lib/styla/types";
import { categoryLabel } from "./ItemCard";
import { Button } from "@/components/ui/button";
import { Loader2, Wand2 } from "lucide-react";
import { startTryOn, checkTryOn } from "@/lib/styla/mock-api";

const AVATARS = [
  { id: "avatar_1", url: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=150&q=80", label: "Model 1" },
  { id: "avatar_2", url: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?auto=format&fit=crop&w=150&q=80", label: "Model 2" },
  { id: "avatar_3", url: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=150&q=80", label: "Model 3" },
  { id: "avatar_4", url: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=150&q=80", label: "Model 4" },
  { id: "avatar_5", url: "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?auto=format&fit=crop&w=150&q=80", label: "Model 5" }
];

export function OutfitCard({ outfit, action }: { outfit: Outfit; action?: React.ReactNode }) {
  const style = STYLES.find((s) => s.id === outfit.style);
  
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0].id);
  const [vtonState, setVtonState] = useState<'idle' | 'loading' | 'done'>('idle');
  const [vtonUrl, setVtonUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  
  useEffect(() => {
    // Polling logic
    let interval: number;
    if (vtonState === 'loading' && jobId) {
      interval = window.setInterval(async () => {
        try {
          const res = await checkTryOn(jobId);
          if (res.status === 'completed') {
            setVtonUrl(res.result_url || null);
            setVtonState('done');
            window.clearInterval(interval);
          } else if (res.status === 'failed') {
            console.error("VTON failed:", res.error);
            setVtonState('idle');
            window.clearInterval(interval);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => {
      if (interval) window.clearInterval(interval);
    };
  }, [vtonState, jobId]);

  const handleTryOn = async () => {
    setVtonState('loading');
    setVtonUrl(null);
    try {
      const jId = await startTryOn(outfit.id, selectedAvatar, outfit.items);
      setJobId(jId);
    } catch (e) {
      console.error(e);
      setVtonState('idle');
    }
  };

  return (
    <article className="glass overflow-hidden rounded-3xl p-4">
      <div className="mb-3 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-display text-xl">{style?.label ?? outfit.style}</h3>
          <p className="text-xs text-muted-foreground">
            {outfit.items.length} pieces ·{" "}
            {new Date(outfit.createdAt).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
            })}
          </p>
        </div>
        {action}
      </div>
      
      {/* VTON Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-white/30 p-3 rounded-2xl mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground mr-2">Avatar:</span>
          <div className="flex gap-2">
            {AVATARS.map(avatar => (
              <button
                key={avatar.id}
                onClick={() => setSelectedAvatar(avatar.id)}
                className={`relative w-8 h-8 rounded-full overflow-hidden border-2 transition-all ${
                  selectedAvatar === avatar.id ? 'border-primary scale-110' : 'border-transparent opacity-70 hover:opacity-100'
                }`}
                title={avatar.label}
              >
                <img src={avatar.url} alt={avatar.label} className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        </div>
        <Button 
          size="sm" 
          className="rounded-full shrink-0" 
          onClick={handleTryOn}
          disabled={vtonState === 'loading'}
        >
          {vtonState === 'loading' ? (
            <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</>
          ) : (
            <><Wand2 className="w-4 h-4 mr-2" /> Try It On</>
          )}
        </Button>
      </div>

      <div className="flex flex-row gap-4 bg-white/50 rounded-2xl p-4 min-h-[320px]">
        {vtonState === 'done' && vtonUrl && (
          <figure className="w-1/2 flex-shrink-0 rounded-xl overflow-hidden shadow-inner bg-black/5 relative animate-in fade-in duration-500">
            <img 
              src={vtonUrl} 
              alt="Virtual Try-On" 
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/60 text-white text-xs rounded backdrop-blur-md">
              Virtual Try-On Result
            </div>
          </figure>
        )}
        
        {vtonState === 'loading' && (
          <div className="w-1/2 flex-shrink-0 rounded-xl shadow-inner bg-black/5 flex flex-col items-center justify-center text-muted-foreground p-4 text-center">
            <Loader2 className="w-8 h-8 animate-spin mb-4 text-primary" />
            <p className="text-sm font-medium">Running CatVTON Pipeline...</p>
            <p className="text-xs opacity-70 mt-1">This uses heavy GPU computation to render garments sequentially.</p>
          </div>
        )}

        <div className={`flex flex-col items-center justify-center -space-y-8 py-4 ${(vtonState === 'done' || vtonState === 'loading') ? 'w-1/2' : 'w-full'}`}>
          {[...outfit.items]
            .sort((a, b) => {
              const order: Record<string, number> = {
                hat: 1, sunglasses: 2, scarf: 3, outerwear: 4, coat: 4, jacket: 5, 
                top: 7, sweater: 6, shirt: 7, 't-shirt': 8, dress: 9, 
                belt: 10, bottom: 12, pants: 11, jeans: 12, skirt: 13, shorts: 14, 
                bag: 15
              };
              return (order[a.category] || 99) - (order[b.category] || 99);
            })
            .map((item, i) => (
            <figure key={item.id} className="relative z-10 w-40 transition-transform hover:scale-110 hover:z-20" style={{ zIndex: 10 - i }}>
              <img
                src={item.imageUrl}
                alt={categoryLabel(item.category)}
                loading="lazy"
                className="w-full object-contain drop-shadow-xl mix-blend-multiply"
              />
            </figure>
          ))}
        </div>
      </div>
    </article>
  );
}
