"use server";

import { redirect } from "next/navigation";
import {
  login,
  register,
  createSession,
  deleteSession,
  getSession,
} from "@/lib/auth";

export interface AuthState {
  error?: string;
  success?: boolean;
}

/**
 * Server action for user login
 */
export async function loginAction(
  _prevState: AuthState | undefined,
  formData: FormData,
): Promise<AuthState> {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  if (!email || !password) {
    return { error: "Email and password are required" };
  }

  try {
    const authResponse = await login(email, password);

    // Decode JWT to get user info (the sub claim contains email)
    const tokenPayload = JSON.parse(
      Buffer.from(authResponse.access_token.split(".")[1], "base64").toString(),
    );

    // Create session with user info (use 1 as default userId since we only have email)
    await createSession(1, tokenPayload.sub, authResponse.access_token);
  } catch (error) {
    return { error: error instanceof Error ? error.message : "Login failed" };
  }

  redirect("/");
}

/**
 * Server action for user registration
 */
export async function registerAction(
  _prevState: AuthState | undefined,
  formData: FormData,
): Promise<AuthState> {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;
  const confirmPassword = formData.get("confirmPassword") as string;

  if (!email || !password) {
    return { error: "Email and password are required" };
  }

  if (password !== confirmPassword) {
    return { error: "Passwords do not match" };
  }

  if (password.length < 6) {
    return { error: "Password must be at least 6 characters" };
  }

  try {
    // Register user
    const user = await register(email, password);

    // Auto-login after registration
    const authResponse = await login(email, password);

    await createSession(user.id, user.email, authResponse.access_token);
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : "Registration failed",
    };
  }

  redirect("/");
}

/**
 * Server action for logout
 */
export async function logoutAction(): Promise<void> {
  await deleteSession();
  redirect("/login");
}

/**
 * Get current session (for client components)
 */
export async function getSessionAction() {
  return getSession();
}
