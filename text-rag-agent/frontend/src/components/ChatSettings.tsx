import * as Popover from "@radix-ui/react-popover";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { SlidersHorizontal } from "lucide-react";

import { getModels, updateSettings } from "@/api/chats";
import type { Chat, ChatSettings as Settings } from "@/types";

type Field = { key: keyof Settings; label: string; min: number; max: number; step: number; ingest?: boolean };

const FIELDS: Field[] = [
  { key: "top_k", label: "Sources retrieved (top-k)", min: 1, max: 20, step: 1 },
  { key: "min_score", label: "Min. similarity to answer", min: 0, max: 1, step: 0.05 },
  { key: "temperature", label: "Answer temperature", min: 0, max: 1, step: 0.1 },
  { key: "history_turns", label: "History turns kept verbatim", min: 0, max: 20, step: 1 },
  { key: "context_token_budget", label: "Context token budget", min: 2000, max: 32000, step: 1000 },
  { key: "chunk_size", label: "Chunk size (new uploads)", min: 256, max: 2048, step: 64, ingest: true },
  { key: "chunk_overlap", label: "Chunk overlap (new uploads)", min: 0, max: 512, step: 32, ingest: true },
];

export function ChatSettings({ chat }: { chat: Chat }) {
  const qc = useQueryClient();
  const { data: modelInfo } = useQuery({ queryKey: ["models"], queryFn: getModels, staleTime: Infinity });
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
          className="z-50 max-h-[80vh] w-72 overflow-y-auto rounded-xl border border-line bg-surface p-4 shadow-lg animate-rise"
        >
          <h3 className="mb-3 font-mono text-xs uppercase tracking-wide text-muted">Retrieval settings</h3>

          <label className="mb-4 block">
            <div className="mb-1 text-xs text-ink">Model</div>
            <select
              value={chat.settings.model}
              onChange={(e) => patch.mutate({ model: e.target.value })}
              className="w-full rounded-md border border-line bg-surface px-2 py-1.5 text-xs text-ink focus:border-primary focus:outline-none"
            >
              {(modelInfo?.models ?? [chat.settings.model]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>

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
            Higher min. similarity makes the assistant refuse more readily. Chunk size/overlap apply only
            to documents uploaded from now on; existing documents keep their chunking.
          </p>
          <Popover.Arrow className="fill-[var(--surface)]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
