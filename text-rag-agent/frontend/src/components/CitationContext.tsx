import { createContext, useContext, useState, type ReactNode } from "react";

interface CitationState {
  activeN: number | null;
  setActiveN: (n: number | null) => void;
}

const Ctx = createContext<CitationState>({ activeN: null, setActiveN: () => {} });

export function CitationProvider({ children }: { children: ReactNode }) {
  const [activeN, setActiveN] = useState<number | null>(null);
  return <Ctx.Provider value={{ activeN, setActiveN }}>{children}</Ctx.Provider>;
}

export const useCitations = () => useContext(Ctx);
