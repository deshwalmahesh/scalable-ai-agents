import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2 } from "lucide-react";

import { deleteChat } from "@/api/chats";
import { cn } from "@/lib/utils";
import type { Chat } from "@/types";

export function ChatList({
  chats,
  activeId,
  onSelect,
  onNew,
}: {
  chats: Chat[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
  onNew: () => void;
}) {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: deleteChat,
    onSuccess: (_d, id) => {
      qc.invalidateQueries({ queryKey: ["chats"] });
      if (id === activeId) onSelect(null);
    },
  });

  return (
    <div className="flex min-h-0 flex-col">
      <button
        onClick={onNew}
        className="mx-3 mb-2 flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium text-ink transition-colors hover:border-primary/50"
      >
        <Plus className="size-4 text-primary" />
        New chat
      </button>
      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2">
        {chats.map((c) => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={cn(
              "group flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
              c.id === activeId ? "bg-surface-2 text-ink" : "text-muted hover:bg-surface-2/60 hover:text-ink",
            )}
          >
            <span className="min-w-0 flex-1 truncate">{c.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                remove.mutate(c.id);
              }}
              aria-label="Delete chat"
              className="grid size-5 shrink-0 place-items-center rounded text-muted opacity-0 hover:text-danger group-hover:opacity-100"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
