import { type CSSProperties, useEffect, useRef } from "react";

import { useCitations } from "@/components/CitationContext";
import { citeVar, cn } from "@/lib/utils";
import type { Citation } from "@/types";

function SourceCard({ c }: { c: Citation }) {
  const { activeN, setActiveN } = useCitations();
  const active = activeN === c.n;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  return (
    <div
      ref={ref}
      onMouseEnter={() => setActiveN(c.n)}
      onMouseLeave={() => setActiveN(null)}
      style={{ ["--c" as string]: citeVar(c.n) } as CSSProperties}
      className={cn(
        "rounded-lg border border-l-[3px] bg-surface p-3 transition-all",
        "border-l-[var(--c)] border-y-line border-r-line",
        active ? "shadow-sm ring-1 ring-[color-mix(in_oklch,var(--c)_40%,transparent)]" : "",
      )}
    >
      <div className="mb-2 flex items-center gap-2">
        <span
          className="grid h-4 min-w-4 place-items-center rounded px-1 font-mono text-[0.65rem] font-semibold"
          style={{
            color: "var(--c)",
            background: "color-mix(in oklch, var(--c) 14%, transparent)",
          }}
        >
          {c.n}
        </span>
        <span className="truncate font-mono text-xs text-ink" title={c.filename}>
          {c.filename}
        </span>
        {c.page != null && <span className="font-mono text-[0.7rem] text-muted">p.{c.page}</span>}
      </div>

      <div className="mb-2 flex items-center gap-2">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-surface-2">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.round(Math.max(0, Math.min(1, c.score)) * 100)}%`,
              background: "var(--c)",
            }}
          />
        </div>
        <span className="font-mono text-[0.7rem] text-muted">{c.score.toFixed(2)}</span>
      </div>

      <p
        className="rounded px-2 py-1.5 text-[0.8rem] leading-relaxed text-ink/90"
        style={{ background: "color-mix(in oklch, var(--c) 9%, transparent)" }}
      >
        {c.snippet}
      </p>
    </div>
  );
}

export function SourcesRail({ citations }: { citations: Citation[] }) {
  return (
    <aside className="hidden w-80 shrink-0 flex-col border-l border-line bg-surface-2/40 lg:flex">
      <div className="border-b border-line px-4 py-3">
        <h2 className="font-mono text-xs uppercase tracking-wide text-muted">Sources</h2>
      </div>
      <div className="flex-1 space-y-2.5 overflow-y-auto p-3">
        {citations.length === 0 ? (
          <p className="px-1 pt-2 text-sm leading-relaxed text-muted">
            Passages used to answer will appear here, colour-matched to the citations in the reply.
          </p>
        ) : (
          citations.map((c) => <SourceCard key={c.n} c={c} />)
        )}
      </div>
    </aside>
  );
}
