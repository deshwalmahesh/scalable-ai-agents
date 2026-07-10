// Same-origin in prod (the API serves the built SPA). In dev set VITE_API_URL=http://localhost:8000.
export const API_BASE = import.meta.env.VITE_API_URL ?? "";

const TOKEN_KEY = "documind_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Pull a human-readable message out of any error body shape the API might return
 *  (string detail, {code, reason}, [validation errors], {error}) with a friendly fallback. */
export function errorMessage(body: unknown, status: number): string {
  const b = (body ?? {}) as { detail?: unknown; error?: unknown; message?: unknown };
  const d = b.detail;
  if (typeof d === "string" && d.trim()) return d;
  if (Array.isArray(d) && (d[0] as { msg?: string })?.msg) return (d[0] as { msg: string }).msg;
  if (d && typeof d === "object") {
    const o = d as { reason?: string; msg?: string };
    if (o.reason) return o.reason;
    if (o.msg) return o.msg;
  }
  if (typeof b.error === "string" && b.error.trim()) return b.error;
  if (typeof b.message === "string" && b.message.trim()) return b.message;
  return status >= 500
    ? "Something went wrong on our end. Please try again."
    : "Something went wrong. Please try again.";
}

export async function apiFetch<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers = new Headers(opts.headers);
  for (const [k, v] of Object.entries(authHeaders())) headers.set(k, v);
  const body = opts.body;
  const skipJson = body instanceof FormData || body instanceof URLSearchParams;
  if (body && !skipJson && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(API_BASE + path, { ...opts, headers });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(errorMessage(body, res.status), res.status);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}
