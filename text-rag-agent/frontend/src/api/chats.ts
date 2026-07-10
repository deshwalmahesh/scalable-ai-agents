import { apiFetch } from "@/api/client";
import type { Chat, ChatSettings, Message } from "@/types";

export function listChats(): Promise<Chat[]> {
  return apiFetch("/chats");
}

export function createChat(title = "New chat"): Promise<Chat> {
  return apiFetch("/chats", { method: "POST", body: JSON.stringify({ title }) });
}

export function deleteChat(id: string): Promise<void> {
  return apiFetch(`/chats/${id}`, { method: "DELETE" });
}

export function getMessages(id: string): Promise<Message[]> {
  return apiFetch(`/chats/${id}/messages`);
}

export function updateSettings(id: string, settings: Partial<ChatSettings>): Promise<Chat> {
  return apiFetch(`/chats/${id}`, { method: "PATCH", body: JSON.stringify(settings) });
}
