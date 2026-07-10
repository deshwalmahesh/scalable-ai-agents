import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-10 w-full rounded-md border border-line bg-surface px-3 text-sm text-ink",
        "placeholder:text-muted focus-visible:border-primary focus-visible:outline-none",
        className,
      )}
      {...props}
    />
  );
}
