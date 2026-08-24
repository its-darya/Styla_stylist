import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export function AnalyzingCard({
  steps,
  className,
}: {
  steps: string[];
  className?: string;
}) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, steps.length - 1)), 800);
    return () => clearInterval(t);
  }, [steps.length]);

  return (
    <div className={cn("glass rounded-3xl p-6", className)}>
      <div className="flex gap-4">
        <div className="shimmer size-24 shrink-0 rounded-2xl bg-muted" />
        <div className="min-w-0 flex-1 space-y-2 py-1">
          <div className="shimmer h-3 w-1/2 rounded-full bg-muted" />
          <div className="shimmer h-3 w-2/3 rounded-full bg-muted" />
          <div className="shimmer h-3 w-1/3 rounded-full bg-muted" />
        </div>
      </div>
      <ul className="mt-5 space-y-1.5 text-sm">
        {steps.map((s, i) => (
          <li
            key={s}
            className={cn(
              "flex items-center gap-2 transition-colors",
              i <= step ? "text-foreground" : "text-muted-foreground/50",
            )}
          >
            <span
              className={cn(
                "size-1.5 rounded-full",
                i < step ? "bg-primary" : i === step ? "animate-pulse bg-primary" : "bg-border",
              )}
            />
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ItemSkeleton() {
  return (
    <div className="glass overflow-hidden rounded-3xl">
      <div className="shimmer aspect-[3/4] bg-muted" />
      <div className="space-y-2 p-3">
        <div className="shimmer h-3 w-2/3 rounded-full bg-muted" />
        <div className="shimmer h-3 w-1/3 rounded-full bg-muted" />
      </div>
    </div>
  );
}
