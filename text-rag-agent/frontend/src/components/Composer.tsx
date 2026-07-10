import { ArrowUp } from "lucide-react";
import { type KeyboardEvent, useRef, useState } from "react";

import { cn } from "@/lib/utils";

export function Composer({
  disabled,
  onSend,
}: {
  disabled: boolean;
  onSend: (question: string) => void;
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
    <div className="border-t border-line bg-paper/80 px-6 py-4 backdrop-blur">
      <div className="mx-auto flex w-full max-w-2xl items-end gap-2 rounded-xl border border-line bg-surface p-2 shadow-sm focus-within:border-primary">
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
          className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-ink placeholder:text-muted focus:outline-none"
        />
        <button
          type="button"
          onClick={send}
          disabled={disabled || !value.trim()}
          aria-label="Send"
          className={cn(
            "grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-primary-fg transition-all",
            "hover:opacity-90 disabled:opacity-40",
          )}
        >
          <ArrowUp className="size-4" />
        </button>
      </div>
    </div>
  );
}
