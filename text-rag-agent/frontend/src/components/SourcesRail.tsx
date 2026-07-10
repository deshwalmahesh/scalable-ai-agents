import { ChevronDown, X } from "lucide-react";
import { type CSSProperties, type RefObject, useEffect, useRef, useState } from "react";

import { useCitations } from "@/components/CitationContext";
import { citeVar, cn } from "@/lib/utils";
import type { Citation } from "@/types";

/** Render the snippet with the answer's used span highlighted brighter and the rest dimmed.
 *  `used` is the exact substring located server-side, so a plain indexOf finds it. */
function Snippet({
  text,
  used,
  markRef,
}: {
  text: string;
  used?: string | null;
  markRef?: RefObject<HTMLElement | null>;
}) {
  const i = used ? text.indexOf(used) : -1;
  if (i === -1 || !used) return <span className="text-ink/90">{text}</span>;
  return (
    <>
      <span className="text-ink/55">{text.slice(0, i)}</span>
      <mark
        ref={markRef as RefObject<HTMLElement>}
        className="rounded-[3px] px-0.5 font-medium text-ink [box-decoration-break:clone]"
        style={{ background: "color-mix(in oklch, var(--c) 24%, transparent)" }}
      >
        {text.slice(i, i + used.length)}
      </mark>
      <span className="text-ink/55">{text.slice(i + used.length)}</span>
    </>
  );
}

function SourceCard({ c }: { c: Citation }) {
  const { activeN, setActiveN } = useCitations();
  const active = activeN === c.n;
  const used = !!c.used_quote;
  const cited = !!c.cited || used; // the answer referenced this source (or a span was located)
  const ref = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const markRef = useRef<HTMLElement>(null);
  // Cited sources open by default; uncited ones collapse (click to expand/collapse).
  const [expanded, setExpanded] = useState(cited);

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  // `cited`/`used_quote` land in later SSE events than the card's first render, so auto-open
  // once the source turns out to be cited — otherwise it stays wrongly collapsed.
  useEffect(() => {
    if (cited) setExpanded(true);
  }, [cited]);

  // When an open card has a match, scroll its snippet box so the first match is in view —
  // confined to the box (set scrollTop) so the whole rail doesn't jump.
  useEffect(() => {
    if (expanded && used && bodyRef.current && markRef.current) {
      bodyRef.current.scrollTop = Math.max(0, markRef.current.offsetTop - 24);
    }
  }, [expanded, used]);

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
      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center gap-2 text-left"
        aria-expanded={expanded}
      >
        <span
          className="grid h-4 min-w-4 place-items-center rounded px-1 font-mono text-[0.65rem] font-semibold"
          style={{ color: "var(--c)", background: "color-mix(in oklch, var(--c) 14%, transparent)" }}
        >
          {c.n}
        </span>
        <span className="truncate font-mono text-xs text-ink" title={c.filename}>
          {c.filename}
        </span>
        {c.page != null && <span className="font-mono text-[0.7rem] text-muted">p.{c.page}</span>}
        {!cited && !expanded && (
          <span className="font-mono text-[0.6rem] uppercase tracking-wide text-muted">not cited</span>
        )}
        <ChevronDown
          className={cn("ml-auto size-3.5 shrink-0 text-muted transition-transform", expanded && "rotate-180")}
        />
      </button>

      <div className="mt-2 flex items-center gap-2">
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

      {expanded && (
        <div
          ref={bodyRef}
          className="relative mt-2 max-h-44 overflow-y-auto rounded px-2 py-2 text-[0.8rem] leading-relaxed"
          style={{ background: "color-mix(in oklch, var(--c) 6%, transparent)" }}
        >
          <Snippet text={c.snippet} used={c.used_quote} markRef={markRef} />
        </div>
      )}
    </div>
  );
}

/** Persistent side panel on wide screens; a toggle-able right drawer below `lg` so the
 *  grounded snippets are always reachable (open/onClose drive the mobile drawer). */
export function SourcesRail({
  citations,
  open = false,
  onClose,
}: {
  citations: Citation[];
  open?: boolean;
  onClose?: () => void;
}) {
  return (
    <>
      {open && (
        <div onClick={onClose} className="fixed inset-0 z-30 bg-ink/25 lg:hidden" aria-hidden="true" />
      )}
      <aside
        className={cn(
          "flex w-96 shrink-0 flex-col border-l border-line bg-surface-2/40",
          // below lg: off-canvas drawer from the right, revealed when `open`
          "max-lg:fixed max-lg:inset-y-0 max-lg:right-0 max-lg:z-40 max-lg:w-[90vw] max-lg:max-w-md",
          "max-lg:shadow-2xl max-lg:transition-transform max-lg:duration-200",
          open ? "max-lg:translate-x-0" : "max-lg:translate-x-full",
        )}
      >
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-line px-4">
          <h2 className="font-mono text-xs uppercase tracking-wide text-muted">
            Sources{citations.length ? ` · ${citations.length}` : ""}
          </h2>
          {onClose && (
            <button
              onClick={onClose}
              aria-label="Close sources"
              className="grid size-7 place-items-center rounded text-muted hover:text-ink lg:hidden"
            >
              <X className="size-4" />
            </button>
          )}
        </div>
        <div className="flex-1 space-y-2.5 overflow-y-auto p-3">
          {citations.length === 0 ? (
            <p className="px-1 pt-2 text-sm leading-relaxed text-muted">
              Passages used to answer will appear here, colour-matched to the citations in the reply.
            </p>
          ) : (
            // Rank: highlighted (has a located span) > cited (no span) > uncited, then by score.
            // Cards keep their [n] so the answer's chips still colour-match.
            [...citations]
              .map((c) => ({ c, rank: c.used_quote ? 2 : c.cited ? 1 : 0 }))
              .sort((a, b) => b.rank - a.rank || b.c.score - a.c.score)
              .map(({ c }) => <SourceCard key={c.n} c={c} />)
          )}
        </div>
      </aside>
    </>
  );
}
