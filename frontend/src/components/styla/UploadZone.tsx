import { useRef, useState, useEffect } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  title: string;
  subtitle: string;
  onFile: (file: File) => void;
  disabled?: boolean;
  className?: string;
}

export function UploadZone({ title, subtitle, onFile, disabled, className }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  useEffect(() => {
    function handlePaste(e: ClipboardEvent) {
      if (disabled) return;
      const file = e.clipboardData?.files?.[0];
      if (file && file.type.startsWith("image/")) {
        onFile(file);
      }
    }
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [disabled, onFile]);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file && !disabled) onFile(file);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
      className={cn(
        "glass flex cursor-pointer flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed border-border/80 px-6 py-12 text-center transition-all",
        over && "border-primary bg-accent-soft",
        disabled && "pointer-events-none opacity-60",
        className,
      )}
    >
      <span className="grid size-12 place-items-center rounded-2xl bg-accent-soft text-primary">
        <UploadCloud className="size-5" />
      </span>
      <div>
        <p className="font-display text-xl">{title}</p>
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
