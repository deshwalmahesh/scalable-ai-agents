import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { deleteDocument, listDocuments, uploadDocuments } from "@/api/documents";
import { cn } from "@/lib/utils";
import type { Doc, DocStatus } from "@/types";

const ACCEPT = ".pdf,.docx,.doc,.txt,.md";

const STATUS: Record<DocStatus, { label: string; className: string; pulse?: boolean }> = {
  pending: { label: "queued", className: "text-muted", pulse: true },
  processing: { label: "reading", className: "text-primary", pulse: true },
  ready: { label: "ready", className: "text-[var(--cite-4)]" },
  failed: { label: "failed", className: "text-danger" },
};

function StatusPill({ status }: { status: DocStatus }) {
  const s = STATUS[status];
  return (
    <span className={cn("flex items-center gap-1 font-mono text-[0.65rem]", s.className)}>
      <span className={cn("size-1.5 rounded-full bg-current", s.pulse && "animate-pulse-soft")} />
      {s.label}
    </span>
  );
}

export function DocPanel() {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    // Poll only while something is still ingesting.
    refetchInterval: (q) =>
      (q.state.data as Doc[] | undefined)?.some((d) => d.status === "pending" || d.status === "processing")
        ? 2000
        : false,
  });

  const upload = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
  const remove = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });

  function pick(files: FileList | null) {
    if (files && files.length) upload.mutate(Array.from(files));
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between px-3 pb-1.5">
        <span className="font-mono text-xs uppercase tracking-wide text-muted">Documents</span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "mx-3 mb-2 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed px-3 py-3 text-xs transition-colors",
          dragging ? "border-primary bg-primary/5 text-primary" : "border-line text-muted hover:border-primary/50",
        )}
      >
        <Upload className="size-3.5" />
        {upload.isPending ? "Uploading…" : "Drop files or browse"}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => pick(e.target.files)}
      />
      {upload.isError && <p className="px-3 pb-1 text-xs text-danger">Upload failed. Check size/type.</p>}

      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
        {docs.map((d) => (
          <div key={d.id} className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-surface-2">
            <div className="min-w-0 flex-1">
              <div className="truncate text-xs text-ink" title={d.filename}>
                {d.filename}
              </div>
              <StatusPill status={d.status} />
            </div>
            <button
              onClick={() => remove.mutate(d.id)}
              aria-label={`Delete ${d.filename}`}
              className="grid size-6 shrink-0 place-items-center rounded text-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
