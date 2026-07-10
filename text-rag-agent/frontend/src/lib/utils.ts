import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** CSS var for citation hue N (cycles through the 5 signature hues). */
export function citeVar(n: number): string {
  return `var(--cite-${(n - 1) % 5})`;
}
