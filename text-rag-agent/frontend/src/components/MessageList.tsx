import { Sparkles } from "lucide-react";
import { useEffect, useRef } from "react";

import { AnswerMessage } from "@/components/AnswerMessage";
import { cn } from "@/lib/utils";
import type { Citation, Message } from "@/types";

export interface LiveTurn {
  question: string;
  text: string;
  grounded: boolean | null;
  streaming: boolean;
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
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
}: {
  messages: Message[];
  live: LiveTurn | null;
  onFocus: (id: string, citations: Citation[]) => void;
  focusedId: string | null;
}) {
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
          <UserBubble key={m.id} text={m.content} />
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
