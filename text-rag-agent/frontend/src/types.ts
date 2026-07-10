export interface User {
  id: string;
  email: string;
  name: string;
  plan: string;
}

export type DocStatus = "pending" | "processing" | "ready" | "failed";

export interface Doc {
  id: string;
  filename: string;
  status: DocStatus;
  chunk_count: number;
  error: string | null;
  created_at: string;
}

export interface ChatSettings {
  top_k: number;
  min_score: number;
  history_turns: number;
  context_token_budget: number;
  temperature: number;
  document_ids: string[] | null;
}

export interface Chat {
  id: string;
  title: string;
  settings: ChatSettings;
  has_summary: boolean;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  n: number;
  document_id: string;
  filename: string;
  page: number | null;
  score: number;
  snippet: string;
}

export interface Verdict {
  source: "context" | "history" | "both" | "none" | string;
  grounded: boolean;
  correct: boolean;
  reason: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  verdict: Verdict | null;
  created_at: string;
}
