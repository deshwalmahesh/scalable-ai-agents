import { API_BASE, authHeaders } from "@/api/client";
import type { Citation, Verdict } from "@/types";

export interface StreamDone {
  message_id: string;
  citations: Citation[];
  grounded: boolean;
}

export type StreamVerdict = Verdict & { message_id: string };

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (done: StreamDone) => void;
  onVerdict?: (verdict: StreamVerdict) => void;
  onError: (message: string) => void;
}

function handleBlock(block: string, h: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (!dataLines.length) return;
  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    data = dataLines.join("\n");
  }
  if (event === "token") h.onToken(typeof data === "string" ? data : String(data));
  else if (event === "done") h.onDone(data as StreamDone);
  else if (event === "verdict") h.onVerdict?.(data as StreamVerdict);
  else if (event === "error") h.onError((data as { message?: string })?.message ?? "Stream failed");
}

/** POST a question and stream the grounded answer (SSE over fetch). */
export async function streamAnswer(
  chatId: string,
  question: string,
  documentIds: string[] | null,
  h: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chats/${chatId}/query`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: documentIds }),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* ignore */
    }
    h.onError(detail);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      handleBlock(buffer.slice(0, sep), h);
      buffer = buffer.slice(sep + 2);
    }
  }
}
