import { apiFetch } from "@/api/client";
import type { Doc } from "@/types";

export function listDocuments(): Promise<Doc[]> {
  return apiFetch("/documents");
}

export function getDocument(id: string): Promise<Doc> {
  return apiFetch(`/documents/${id}`);
}

export function uploadDocuments(files: File[]): Promise<Doc[]> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return apiFetch("/documents", { method: "POST", body: form });
}

export function deleteDocument(id: string): Promise<void> {
  return apiFetch(`/documents/${id}`, { method: "DELETE" });
}
