import { useAuth } from "@/hooks/useAuth";
import { AuthPage } from "@/pages/Auth";
import { ChatPage } from "@/pages/Chat";

export function App() {
  const { user, ready } = useAuth();
  if (!ready) {
    return <div className="grid h-screen place-items-center font-mono text-sm text-muted">Loading…</div>;
  }
  return user ? <ChatPage /> : <AuthPage />;
}
