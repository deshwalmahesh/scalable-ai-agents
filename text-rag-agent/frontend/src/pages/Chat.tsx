import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LogOut, PanelRight, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { createChat, deleteLastExchange, getMessages, listChats } from "@/api/chats";
import { streamAnswer } from "@/api/stream";
import { ChatList } from "@/components/ChatList";
import { ChatSettings } from "@/components/ChatSettings";
import { CitationProvider } from "@/components/CitationContext";
import { Composer } from "@/components/Composer";
import { DocsMenu } from "@/components/DocsMenu";
import { Logo } from "@/components/Logo";
import { type LiveTurn, MessageList } from "@/components/MessageList";
import { SourcesRail } from "@/components/SourcesRail";
import { useAuth } from "@/hooks/useAuth";
import type { Chat, Citation, Message } from "@/types";

function nowIso() {
  return new Date().toISOString();
}

export function ChatPage() {
  const qc = useQueryClient();
  const { user, logout } = useAuth();

  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [focused, setFocused] = useState<{ id: string | null; citations: Citation[] }>({
    id: null,
    citations: [],
  });
  const [error, setError] = useState<string | null>(null);
  const [showSources, setShowSources] = useState(false);
  const editingRef = useRef(false);
  const textRef = useRef("");

  /** Drop the last user turn and everything after it (its answer) from a message list. */
  function dropLastExchange(msgs: Message[]): Message[] {
    let i = msgs.length - 1;
    while (i >= 0 && msgs[i].role !== "user") i--;
    return i >= 0 ? msgs.slice(0, i) : msgs;
  }

  /** Inline edit of the last message: mark it as an edit, then re-ask with the new text. */
  function submitEdit(content: string) {
    editingRef.current = true;
    handleSend(content);
  }

  const { data: chats = [] } = useQuery({ queryKey: ["chats"], queryFn: listChats });
  const activeChat: Chat | undefined = chats.find((c) => c.id === activeChatId);

  const { data: messages = [] } = useQuery({
    queryKey: ["messages", activeChatId],
    queryFn: () => getMessages(activeChatId as string),
    enabled: !!activeChatId,
  });

  const newChat = useMutation({
    mutationFn: (title: string) => createChat(title),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["chats"] });
      setActiveChatId(c.id);
    },
  });

  // Show the newest answer's sources in the rail (unless a turn is streaming).
  useEffect(() => {
    if (live) return;
    const lastAnswer = [...messages].reverse().find((m) => m.role === "assistant");
    setFocused({ id: lastAnswer?.id ?? null, citations: lastAnswer?.citations ?? [] });
  }, [messages, live]);

  function selectChat(id: string | null) {
    setActiveChatId(id);
    setLive(null);
    setError(null);
    setFocused({ id: null, citations: [] });
    editingRef.current = false;
  }

  async function handleSend(question: string) {
    setError(null);
    let chatId = activeChatId;
    // Editing the last message: remove that turn + its answer, then re-ask with the new text.
    if (editingRef.current && chatId) {
      editingRef.current = false;
      try {
        await deleteLastExchange(chatId);
      } catch {
        /* if it's already gone, the re-ask below still works */
      }
      qc.setQueryData<Message[]>(["messages", chatId], (old = []) => dropLastExchange(old));
    }
    if (!chatId) {
      const c = await newChat.mutateAsync(question.slice(0, 48));
      chatId = c.id;
    }
    const cid = chatId;
    textRef.current = "";
    setLive({ question, text: "", grounded: null, streaming: true });
    setFocused({ id: null, citations: [] });

    await streamAnswer(cid, question, null, {
      onToken: (t) => {
        textRef.current += t;
        setLive((l) => (l ? { ...l, text: textRef.current } : l));
      },
      onDone: (d) => {
        const answer = textRef.current;
        qc.setQueryData<Message[]>(["messages", cid], (old = []) => [
          ...old,
          { id: `u-${cid}-${old.length}`, role: "user", content: question, citations: null, verdict: null, created_at: nowIso() },
          { id: d.message_id, role: "assistant", content: answer, citations: d.citations, verdict: null, created_at: nowIso() },
        ]);
        setFocused({ id: d.message_id, citations: d.citations });
        setLive(null);
        qc.invalidateQueries({ queryKey: ["chats"] });
      },
      onVerdict: (v) => {
        // Arrives just after `done`: attach the verdict and the enriched citations (which now
        // carry used_quote spans for line-level highlighting) to the finalized message.
        const verdict = { source: v.source, grounded: v.grounded, correct: v.correct, reason: v.reason };
        qc.setQueryData<Message[]>(["messages", cid], (old = []) =>
          old.map((m) =>
            m.id === v.message_id ? { ...m, verdict, ...(v.citations ? { citations: v.citations } : {}) } : m,
          ),
        );
        if (v.citations) {
          setFocused((f) => (f.id === v.message_id ? { ...f, citations: v.citations! } : f));
        }
      },
      onError: (msg) => {
        setError(msg);
        setLive(null);
        if (activeChatId) qc.invalidateQueries({ queryKey: ["messages", activeChatId] });
      },
    });
  }

  return (
    <CitationProvider>
      <div className="flex h-screen bg-paper text-ink">
        {/* Left rail */}
        <aside className="flex w-64 shrink-0 flex-col border-r border-line bg-surface-2/40">
          <div className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-4">
            <Logo size={22} />
            <span className="font-serif text-lg font-semibold tracking-tight">DocuMind</span>
          </div>
          <div className="flex min-h-0 flex-1 flex-col gap-4 pb-2">
            <ChatList
              chats={chats}
              activeId={activeChatId}
              onSelect={selectChat}
              onNew={() => selectChat(null)}
            />
          </div>
          <div className="flex items-center justify-between border-t border-line px-3 py-2.5">
            <span className="truncate text-xs text-muted" title={user?.email}>
              {user?.email}
            </span>
            <button
              onClick={logout}
              aria-label="Sign out"
              className="grid size-7 shrink-0 place-items-center rounded text-muted hover:text-ink"
            >
              <LogOut className="size-4" />
            </button>
          </div>
        </aside>

        {/* Center */}
        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-14 shrink-0 items-center justify-between border-b border-line px-6">
            <h1 className="truncate font-serif text-base text-ink">
              {activeChat?.title ?? "New chat"}
            </h1>
            <div className="flex items-center gap-1">
              <DocsMenu chat={activeChat} />
              {activeChat && <ChatSettings chat={activeChat} />}
              <button
                onClick={() => setShowSources(true)}
                aria-label="Show sources"
                className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted hover:bg-surface-2 hover:text-ink lg:hidden"
              >
                <PanelRight className="size-4" />
                <span className="font-mono text-xs">
                  Sources{focused.citations.length ? ` · ${focused.citations.length}` : ""}
                </span>
              </button>
            </div>
          </header>

          {error && (
            <div className="mx-6 mt-3 flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">
              <span className="flex-1">{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss">
                <X className="size-4" />
              </button>
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            <MessageList
              messages={messages}
              live={live}
              focusedId={focused.id}
              onFocus={(id, citations) => setFocused({ id, citations })}
              onEditSubmit={submitEdit}
            />
          </div>

          <Composer disabled={!!live?.streaming} onSend={handleSend} chat={activeChat} />
        </main>

        {/* Right rail — persistent on wide screens, toggle-able drawer below lg */}
        <SourcesRail
          citations={focused.citations}
          open={showSources}
          onClose={() => setShowSources(false)}
        />
      </div>
    </CitationProvider>
  );
}
