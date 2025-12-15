/**
 * Authentication utilities for session management
 */
import { SignJWT, jwtVerify } from 'jose';
import { cookies } from 'next/headers';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';
const SESSION_SECRET = new TextEncoder().encode(
  process.env.SESSION_SECRET || 'your-secret-key-min-32-characters-long!'
);

export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface Session {
  userId: number;
  email: string;
  accessToken: string;
  expiresAt: Date;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

/**
 * Register a new user
 */
export async function register(email: string, password: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(error.detail || 'Registration failed');
  }

  return response.json();
}

/**
 * Login user and get access token
 */
export async function login(email: string, password: string): Promise<AuthResponse> {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData.toString(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(error.detail || 'Invalid email or password');
  }

  return response.json();
}

/**
 * Create encrypted session cookie
 */
export async function createSession(userId: number, email: string, accessToken: string) {
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000); // 7 days

  const session = await new SignJWT({ userId, email, accessToken })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime(expiresAt)
    .sign(SESSION_SECRET);

  const cookieStore = await cookies();
  cookieStore.set('session', session, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    expires: expiresAt,
    sameSite: 'lax',
    path: '/',
  });
}

/**
 * Decrypt and verify session from cookie
 */
export async function getSession(): Promise<Session | null> {
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get('session')?.value;

  if (!sessionCookie) return null;

  try {
    const { payload } = await jwtVerify(sessionCookie, SESSION_SECRET);
    return {
      userId: payload.userId as number,
      email: payload.email as string,
      accessToken: payload.accessToken as string,
      expiresAt: new Date((payload.exp as number) * 1000),
    };
  } catch {
    return null;
  }
}

/**
 * Decrypt session (for proxy use)
 */
export async function decrypt(cookie: string | undefined): Promise<Session | null> {
  if (!cookie) return null;

  try {
    const { payload } = await jwtVerify(cookie, SESSION_SECRET);
    return {
      userId: payload.userId as number,
      email: payload.email as string,
      accessToken: payload.accessToken as string,
      expiresAt: new Date((payload.exp as number) * 1000),
    };
  } catch {
    return null;
  }
}

/**
 * Delete session cookie (logout)
 */
export async function deleteSession() {
  const cookieStore = await cookies();
  cookieStore.delete('session');
}

/**
 * Get current user from session
 */
export async function getCurrentUser(): Promise<{ userId: number; email: string } | null> {
  const session = await getSession();
  if (!session) return null;
  return { userId: session.userId, email: session.email };
}

/**
 * Get access token from session
 */
export async function getAccessToken(): Promise<string | null> {
  const session = await getSession();
  return session?.accessToken ?? null;
}
