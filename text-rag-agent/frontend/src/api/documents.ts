import { apiFetch } from "@/api/client";
import type { Doc } from "@/types";

export function listDocuments(): Promise<Doc[]> {
  return apiFetch("/documents");
}

export function getDocument(id: string): Promise<Doc> {
  return apiFetch(`/documents/${id}`);
}

export function uploadDocuments(
  files: File[],
  opts?: { chunk_size?: number; chunk_overlap?: number },
): Promise<Doc[]> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const q = new URLSearchParams();
  if (opts?.chunk_size) q.set("chunk_size", String(opts.chunk_size));
  if (opts?.chunk_overlap != null) q.set("chunk_overlap", String(opts.chunk_overlap));
  const qs = q.toString();
  return apiFetch(`/documents${qs ? `?${qs}` : ""}`, { method: "POST", body: form });
}

export function deleteDocument(id: string): Promise<void> {
  return apiFetch(`/documents/${id}`, { method: "DELETE" });
}
