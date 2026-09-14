import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { configureApiAuth, fetchMe, login as apiLogin, signup as apiSignup } from "./api";
import type { User } from "./types";

const TOKEN_KEY = "styla.token";
const USER_KEY = "styla.user";

export type AuthStatus = "loading" | "signed-out" | "signed-in";

interface AuthState {
  status: AuthStatus;
  user: User | null;
  token: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

function readStorage(): { token: string | null; user: User | null } {
  if (typeof window === "undefined") return { token: null, user: null };
  try {
    const token = window.localStorage.getItem(TOKEN_KEY);
    const rawUser = window.localStorage.getItem(USER_KEY);
    return { token, user: rawUser ? (JSON.parse(rawUser) as User) : null };
  } catch {
    return { token: null, user: null };
  }
}

function writeStorage(token: string | null, user: User | null) {
  if (typeof window === "undefined") return;
  try {
    if (token && user) {
      window.localStorage.setItem(TOKEN_KEY, token);
      window.localStorage.setItem(USER_KEY, JSON.stringify(user));
    } else {
      window.localStorage.removeItem(TOKEN_KEY);
      window.localStorage.removeItem(USER_KEY);
    }
  } catch {
    /* storage unavailable (private mode) — session lives in memory only */
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const tokenRef = useRef<string | null>(null);

  const applySession = useCallback((nextToken: string | null, nextUser: User | null) => {
    tokenRef.current = nextToken;
    setToken(nextToken);
    setUser(nextUser);
    setStatus(nextToken && nextUser ? "signed-in" : "signed-out");
    writeStorage(nextToken, nextUser);
  }, []);

  const signOut = useCallback(() => applySession(null, null), [applySession]);

  // Register the token source with the API client once.
  useEffect(() => {
    configureApiAuth(() => tokenRef.current, signOut);
  }, [signOut]);

  // Restore the session on the client, then verify it with the backend.
  useEffect(() => {
    const stored = readStorage();
    if (!stored.token || !stored.user) {
      applySession(null, null);
      return;
    }
    applySession(stored.token, stored.user);
    let alive = true;
    fetchMe()
      .then((fresh) => {
        if (alive) applySession(stored.token, fresh);
      })
      .catch(() => {
        /* network error: keep the cached session; a 401 already signed us out */
      });
    return () => {
      alive = false;
    };
  }, [applySession]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const res = await apiLogin(email, password);
      applySession(res.token, res.user);
    },
    [applySession],
  );

  const signUp = useCallback(
    async (email: string, password: string, name: string) => {
      const res = await apiSignup(email, password, name);
      applySession(res.token, res.user);
    },
    [applySession],
  );

  const value = useMemo<AuthState>(
    () => ({ status, user, token, signIn, signUp, signOut }),
    [status, user, token, signIn, signUp, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
