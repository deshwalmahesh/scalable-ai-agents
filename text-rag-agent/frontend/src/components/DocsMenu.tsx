import * as Popover from "@radix-ui/react-popover";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Trash2, Upload } from "lucide-react";
import { type ReactNode, useRef, useState } from "react";

import { updateSettings } from "@/api/chats";
import { deleteDocument, listDocuments, uploadDocuments } from "@/api/documents";
import { cn } from "@/lib/utils";
import type { Chat, Doc, DocStatus } from "@/types";

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
    <span className={cn("flex items-center gap-1 font-mono text-[0.62rem]", s.className)}>
      <span className={cn("size-1.5 rounded-full bg-current", s.pulse && "animate-pulse-soft")} />
      {s.label}
    </span>
  );
}

/** Per-user document library + this chat's scope. Documents are uploaded/embedded once and
 *  reused across chats; each chat chooses which of them it queries (checkbox = in this chat,
 *  default = all). Lives in the chat header so scope is always in the chat's context. */
export function DocsMenu({
  chat,
  trigger,
  side = "bottom",
  align = "end",
}: {
  chat: Chat | undefined;
  trigger?: ReactNode;
  side?: "top" | "bottom";
  align?: "start" | "end";
}) {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    refetchInterval: (q) =>
      (q.state.data as Doc[] | undefined)?.some((d) => d.status === "pending" || d.status === "processing")
        ? 2000
        : false,
  });

  const upload = useMutation({
    mutationFn: (files: File[]) =>
      uploadDocuments(
        files,
        chat ? { chunk_size: chat.settings.chunk_size, chunk_overlap: chat.settings.chunk_overlap } : undefined,
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
  const remove = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
  const patchScope = useMutation({
    mutationFn: (document_ids: string[] | null) => updateSettings(chat!.id, { document_ids }),
    // Optimistic: update the cache immediately so rapid toggles read fresh scope and render at
    // once (otherwise a quick second toggle reads stale document_ids and clobbers the first).
    onMutate: (document_ids) => {
      qc.setQueryData<Chat[]>(["chats"], (old = []) =>
        old.map((c) => (c.id === chat!.id ? { ...c, settings: { ...c.settings, document_ids } } : c)),
      );
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ["chats"] }),
  });

  const allIds = docs.map((d) => d.id);
  const scope = chat?.settings.document_ids ?? null; // null = all docs
  const inChat = (id: string) => scope === null || scope.includes(id);
  const activeCount = scope === null ? allIds.length : scope.filter((id) => allIds.includes(id)).length;

  function toggle(id: string) {
    if (!chat) return;
    const next = new Set(scope ?? allIds);
    next.has(id) ? next.delete(id) : next.add(id);
    const arr = allIds.filter((x) => next.has(x));
    // Normalize "everything selected" → null so newly uploaded docs are auto-included.
    patchScope.mutate(arr.length === allIds.length ? null : arr);
  }

  function pick(files: FileList | null) {
    if (files && files.length) upload.mutate(Array.from(files));
  }

  const label = !docs.length ? "Docs" : scope === null ? `Docs · all ${allIds.length}` : `Docs · ${activeCount}/${allIds.length}`;

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        {trigger ?? (
          <button
            aria-label="Documents"
            className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted transition-colors hover:bg-surface-2 hover:text-ink"
          >
            <FileText className="size-4" />
            <span className="font-mono text-xs">{label}</span>
          </button>
        )}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side={side}
          align={align}
          sideOffset={8}
          className="z-50 flex max-h-[70vh] w-80 flex-col rounded-xl border border-line bg-surface p-3 shadow-lg animate-rise"
        >
          <div className="mb-2 flex items-center justify-between px-1">
            <h3 className="font-mono text-xs uppercase tracking-wide text-muted">Documents</h3>
            {chat && docs.length > 0 && (
              <span className="font-mono text-[0.65rem] text-muted">
                {activeCount === allIds.length ? "all in this chat" : `${activeCount} in this chat`}
              </span>
            )}
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
              "mb-2 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed px-3 py-3 text-xs transition-colors",
              dragging
                ? "border-primary bg-primary/5 text-primary"
                : "border-line text-muted hover:border-primary/50",
            )}
          >
            <Upload className="size-3.5" />
            {upload.isPending ? "Uploading…" : "Drop files or browse to add to your library"}
          </div>
          <input ref={inputRef} type="file" accept={ACCEPT} multiple hidden onChange={(e) => pick(e.target.files)} />
          {upload.isError && <p className="px-1 pb-1 text-xs text-danger">Upload failed. Check size/type.</p>}

          <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
            {docs.length === 0 ? (
              <p className="px-1 py-2 text-xs leading-relaxed text-muted">
                No documents yet. Upload one, then ask questions about it.
              </p>
            ) : (
              docs.map((d) => (
                <div key={d.id} className="group flex items-center gap-2 rounded-md px-1.5 py-1.5 hover:bg-surface-2">
                  <input
                    type="checkbox"
                    checked={inChat(d.id)}
                    disabled={!chat}
                    onChange={() => toggle(d.id)}
                    title={chat ? "Use in this chat" : "Open a chat to scope documents"}
                    className="size-3.5 shrink-0 accent-[var(--primary)]"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs text-ink" title={d.filename}>
                      {d.filename}
                    </div>
                    <StatusPill status={d.status} />
                  </div>
                  <button
                    onClick={() => remove.mutate(d.id)}
                    aria-label={`Delete ${d.filename}`}
                    className="grid size-6 shrink-0 place-items-center rounded text-muted opacity-0 hover:text-danger group-hover:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>

          {chat && docs.length > 0 && (
            <p className="mt-2 border-t border-line px-1 pt-2 text-[0.65rem] leading-relaxed text-muted">
              Checked documents are searched for answers in this chat. Uploaded once, reused across chats.
            </p>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
