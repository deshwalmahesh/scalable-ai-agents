import { ArrowUp, Paperclip } from "lucide-react";
import { type KeyboardEvent, useRef, useState } from "react";

import { DocsMenu } from "@/components/DocsMenu";
import { cn } from "@/lib/utils";
import type { Chat } from "@/types";

export function Composer({
  disabled,
  onSend,
  chat,
}: {
  disabled: boolean;
  onSend: (question: string) => void;
  chat: Chat | undefined;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  function grow() {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  function send() {
    const q = value.trim();
    if (!q || disabled) return;
    onSend(q);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="px-6 pb-5 pt-1">
      <div className="mx-auto flex w-full max-w-2xl items-end gap-2 rounded-[1.4rem] border border-line bg-surface px-3 py-2 shadow-sm transition-all focus-within:border-primary/70 focus-within:shadow-md">
        <DocsMenu
          chat={chat}
          side="top"
          align="start"
          trigger={
            <button
              type="button"
              aria-label="Attach documents"
              className="grid size-8 shrink-0 place-items-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-ink"
            >
              <Paperclip className="size-4" />
            </button>
          }
        />
        <textarea
          ref={ref}
          rows={1}
          value={value}
          disabled={disabled}
          onChange={(e) => {
            setValue(e.target.value);
            grow();
          }}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about your documents…"
          className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-1.5 text-sm text-ink placeholder:text-muted focus:outline-none focus-visible:outline-none"
        />
        <button
          type="button"
          onClick={send}
          disabled={disabled || !value.trim()}
          aria-label="Send"
          className={cn(
            "grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-fg transition-all",
            "hover:opacity-90 disabled:opacity-40",
          )}
        >
          <ArrowUp className="size-4" />
        </button>
      </div>
    </div>
  );
}
