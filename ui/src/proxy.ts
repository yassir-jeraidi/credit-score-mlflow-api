import { type NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";

const SESSION_SECRET = new TextEncoder().encode(
  process.env.SESSION_SECRET || "your-secret-key-min-32-characters-long!",
);

interface Session {
  userId: number;
  email: string;
  accessToken: string;
  expiresAt: Date;
}

/**
 * Decrypt session cookie (inline for Edge runtime compatibility)
 */
async function decryptSession(
  cookie: string | undefined,
): Promise<Session | null> {
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

// Routes accessible without authentication
const publicRoutes = ["/login", "/register"];

export default async function proxy(req: NextRequest) {
  const path = req.nextUrl.pathname;

  // Check if it's a public route (login, register)
  const isPublicRoute = publicRoutes.some(
    (route) => path === route || path.startsWith(`${route}/`),
  );

  // Get and decrypt session from cookie
  const cookie = req.cookies.get("session")?.value;
  const session = await decryptSession(cookie);
  const isAuthenticated = session !== null && session.email !== undefined;

  // Redirect to home if accessing auth pages while logged in
  if (isPublicRoute && isAuthenticated) {
    return NextResponse.redirect(new URL("/", req.nextUrl));
  }

  // Redirect to login if accessing any non-public route without session
  if (!isPublicRoute && !isAuthenticated) {
    return NextResponse.redirect(new URL("/login", req.nextUrl));
  }

  return NextResponse.next();
}

// Configure which routes the proxy runs on
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - api routes (handled separately)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico, sitemap.xml, robots.txt (metadata files)
     * - public assets (images, etc.)
     */
    "/((?!api|_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt|.*\\.png$|.*\\.svg$).*)",
  ],
};
