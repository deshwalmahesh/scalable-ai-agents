import { Check, Pencil, Sparkles, X } from "lucide-react";
import { type KeyboardEvent, useEffect, useRef, useState } from "react";

import { AnswerMessage } from "@/components/AnswerMessage";
import { cn } from "@/lib/utils";
import type { Citation, Message } from "@/types";

export interface LiveTurn {
  question: string;
  text: string;
  grounded: boolean | null;
  streaming: boolean;
}

function UserBubble({ text, onEditSubmit }: { text: string; onEditSubmit?: (t: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(text);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!editing) return;
    setValue(text);
    const el = ref.current;
    if (el) {
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }
  }, [editing, text]);

  function save() {
    const t = value.trim();
    if (!t) return;
    setEditing(false);
    onEditSubmit?.(t);
  }
  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      save();
    } else if (e.key === "Escape") {
      setEditing(false);
    }
  }

  if (editing) {
    return (
      <div className="flex justify-end">
        <div className="w-full max-w-[85%] rounded-2xl rounded-br-sm border border-primary/50 bg-surface p-2 shadow-sm">
          <textarea
            ref={ref}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onKeyDown={onKeyDown}
            className="max-h-60 w-full resize-none bg-transparent px-2 py-1 text-[0.95rem] text-ink focus:outline-none"
          />
          <div className="mt-1 flex items-center justify-end gap-1">
            <button
              onClick={() => setEditing(false)}
              aria-label="Cancel edit"
              title="Cancel (Esc)"
              className="grid size-7 place-items-center rounded-md text-muted hover:bg-surface-2 hover:text-ink"
            >
              <X className="size-4" />
            </button>
            <button
              onClick={save}
              disabled={!value.trim()}
              aria-label="Save and resend"
              title="Save & resend (Enter)"
              className="grid size-7 place-items-center rounded-md bg-primary text-primary-fg disabled:opacity-40"
            >
              <Check className="size-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex items-center justify-end gap-1.5">
      {onEditSubmit && (
        <button
          onClick={() => setEditing(true)}
          aria-label="Edit message"
          title="Edit — replaces this message and its answer"
          className="grid size-6 shrink-0 place-items-center rounded text-muted opacity-0 transition-opacity hover:text-ink group-hover:opacity-100"
        >
          <Pencil className="size-3.5" />
        </button>
      )}
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-surface-2 px-4 py-2.5 text-[0.95rem] text-ink">
        {text}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="grid flex-1 place-items-center py-20 text-center animate-rise">
      <div className="max-w-xs">
        <Sparkles className="mx-auto mb-4 size-6 text-primary" />
        <h2 className="font-serif text-xl text-ink">Ask your documents</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Upload a file in the sidebar, wait for it to turn ready, then ask a question. Answers cite the
          exact passages they came from.
        </p>
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  live,
  onFocus,
  focusedId,
  onEditSubmit,
}: {
  messages: Message[];
  live: LiveTurn | null;
  onFocus: (id: string, citations: Citation[]) => void;
  focusedId: string | null;
  onEditSubmit?: (content: string) => void;
}) {
  // The last user turn is editable in place — but not while a new answer is streaming.
  const lastUserId = live ? null : [...messages].reverse().find((m) => m.role === "user")?.id;
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, live?.text, live?.streaming]);

  if (messages.length === 0 && !live) {
    return (
      <div className="flex flex-1 flex-col">
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-6 py-8">
      {messages.map((m) =>
        m.role === "user" ? (
          <UserBubble
            key={m.id}
            text={m.content}
            onEditSubmit={m.id === lastUserId ? onEditSubmit : undefined}
          />
        ) : (
          <div
            key={m.id}
            onClick={() => onFocus(m.id, m.citations ?? [])}
            className={cn(
              "cursor-pointer rounded-lg px-3 py-2 -mx-3 animate-rise transition-colors",
              focusedId === m.id ? "bg-surface-2/60" : "hover:bg-surface-2/40",
            )}
          >
            <AnswerMessage
              content={m.content}
              citations={m.citations ?? []}
              grounded={(m.citations?.length ?? 0) > 0}
              verdict={m.verdict}
            />
          </div>
        ),
      )}

      {live && (
        <>
          <UserBubble text={live.question} />
          <div className="px-3 py-2 -mx-3">
            <AnswerMessage
              content={live.text}
              grounded={live.grounded}
              streaming={live.streaming}
            />
          </div>
        </>
      )}
      <div ref={endRef} />
    </div>
  );
}
