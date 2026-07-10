import * as Popover from "@radix-ui/react-popover";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";

import { updateSettings } from "@/api/chats";
import type { Chat, ChatSettings as Settings } from "@/types";

const FIELDS: { key: keyof Settings; label: string; min: number; max: number; step: number }[] = [
  { key: "top_k", label: "Sources retrieved (top-k)", min: 1, max: 20, step: 1 },
  { key: "min_score", label: "Min. similarity to answer", min: 0, max: 1, step: 0.05 },
  { key: "temperature", label: "Answer temperature", min: 0, max: 1, step: 0.1 },
];

export function ChatSettings({ chat }: { chat: Chat }) {
  const qc = useQueryClient();
  const patch = useMutation({
    mutationFn: (s: Partial<Settings>) => updateSettings(chat.id, s),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chats"] }),
  });

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          aria-label="Chat settings"
          className="grid size-8 place-items-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <SlidersHorizontal className="size-4" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          className="z-50 w-72 rounded-xl border border-line bg-surface p-4 shadow-lg animate-rise"
        >
          <h3 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted">Retrieval settings</h3>
          <div className="space-y-4">
            {FIELDS.map((f) => (
              <label key={f.key} className="block">
                <div className="mb-1 flex items-center justify-between text-xs text-ink">
                  <span>{f.label}</span>
                  <span className="font-mono text-muted">{chat.settings[f.key] as number}</span>
                </div>
                <input
                  type="range"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  defaultValue={chat.settings[f.key] as number}
                  onChange={(e) => patch.mutate({ [f.key]: Number(e.target.value) })}
                  className="w-full accent-[var(--primary)]"
                />
              </label>
            ))}
          </div>
          <p className="mt-3 text-[0.7rem] leading-relaxed text-muted">
            Higher min. similarity makes the assistant refuse more readily when documents don't cover the
            question.
          </p>
          <Popover.Arrow className="fill-[var(--surface)]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
