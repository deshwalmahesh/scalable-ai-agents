import { FileText, ShieldCheck, TriangleAlert } from "lucide-react";
import { type CSSProperties } from "react";

import { useCitations } from "@/components/CitationContext";
import { citeVar, cn } from "@/lib/utils";
import type { Citation, Verdict } from "@/types";

function CitationChip({ n }: { n: number }) {
  const { activeN, setActiveN } = useCitations();
  const active = activeN === n;
  return (
    <button
      type="button"
      onMouseEnter={() => setActiveN(n)}
      onMouseLeave={() => setActiveN(null)}
      style={{ ["--c" as string]: citeVar(n), color: "var(--c)" } as CSSProperties}
      className={cn(
        "mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded px-1 align-super",
        "font-mono text-[0.65em] font-semibold transition-all",
        "border border-[color-mix(in_oklch,var(--c)_45%,transparent)]",
        active
          ? "bg-[color-mix(in_oklch,var(--c)_22%,transparent)]"
          : "bg-[color-mix(in_oklch,var(--c)_10%,transparent)]",
      )}
    >
      {n}
    </button>
  );
}

const SOURCE_LABEL: Record<string, string> = {
  context: "documents",
  history: "conversation",
  both: "documents + conversation",
};

/** Realtime LLM-as-Judge result: where the answer came from and whether it holds up. */
function VerdictBadge({ verdict }: { verdict: Verdict }) {
  if (verdict.source === "none") return null; // refusals are shown by the refusal card
  const clean = verdict.grounded && verdict.correct;
  const where = SOURCE_LABEL[verdict.source] ?? verdict.source;
  const label = clean
    ? `Grounded in ${where}`
    : verdict.grounded
      ? `Grounded in ${where}, but may be incomplete`
      : "Unverified — may contain unsupported claims";
  const Icon = clean ? ShieldCheck : TriangleAlert;
  return (
    <div
      title={verdict.reason}
      className={cn(
        "mt-2 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5",
        "font-mono text-[0.68rem] tracking-tight",
        clean
          ? "border-primary/25 bg-primary/5 text-primary"
          : "border-danger/30 bg-danger/5 text-danger",
      )}
    >
      <Icon className="size-3" />
      {label}
    </div>
  );
}

export function AnswerMessage({
  content,
  citations,
  grounded,
  streaming,
  verdict,
}: {
  content: string;
  citations?: Citation[];
  grounded: boolean | null;
  streaming?: boolean;
  verdict?: Verdict | null;
}) {
  if (grounded === false) {
    return (
      <div className="flex items-start gap-3 rounded-lg border border-line bg-surface-2 px-4 py-3">
        <FileText className="mt-0.5 size-4 shrink-0 text-muted" />
        <p className="font-serif text-[0.95rem] leading-relaxed text-muted">
          {content || "Nothing in your documents covers this."}
        </p>
      </div>
    );
  }

  // Known citation numbers (null while streaming, before `done` delivers them). An orphan [n]
  // the model emitted for a source that wasn't retrieved is dropped, not shown as a dead chip.
  const validNs = citations ? new Set(citations.map((c) => c.n)) : null;
  const parts = content.split(/(\[\d+\])/g);
  return (
    <div>
      <div className="font-serif text-[1.02rem] leading-[1.75] text-ink whitespace-pre-wrap [text-wrap:pretty]">
        {parts.map((part, i) => {
          const m = part.match(/^\[(\d+)\]$/);
          if (m) {
            const n = Number(m[1]);
            if (validNs && !validNs.has(n)) return null; // orphan citation → drop
            return <CitationChip key={i} n={n} />;
          }
          return <span key={i}>{part}</span>;
        })}
        {streaming && (
          <span className="ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[0.15em] animate-pulse-soft bg-primary" />
        )}
      </div>
      {!streaming && verdict && <VerdictBadge verdict={verdict} />}
    </div>
  );
}
