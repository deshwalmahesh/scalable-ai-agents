import { apiFetch } from "@/api/client";
import type { User } from "@/types";

export function register(email: string, password: string): Promise<User> {
  return apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function login(email: string, password: string): Promise<string> {
  const body = new URLSearchParams({ username: email, password });
  const res = await apiFetch<{ access_token: string }>("/auth/jwt/login", {
    method: "POST",
    body,
  });
  return res.access_token;
}

export function me(): Promise<User> {
  return apiFetch("/users/me");
}
