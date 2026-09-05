import { NextRequest, NextResponse } from "next/server";

import { getAuth } from "@/lib/auth";
import { getServerEnvironment } from "@/lib/env.server";
import { decideAdminAccess } from "@/lib/auth/policy";

export async function proxy(request: NextRequest) {
  try {
    const environment = getServerEnvironment();
    const session = await getAuth().api.getSession({
      headers: request.headers,
    });
    const decision = decideAdminAccess(
      session,
      environment.ADMIN_EMAIL,
    );

    if (decision === "authorized") {
      return NextResponse.next();
    }

    if (decision === "unauthenticated") {
      return NextResponse.redirect(
        new URL("/sign-in", environment.BETTER_AUTH_URL),
      );
    }

    return NextResponse.redirect(
      new URL("/access-denied", environment.BETTER_AUTH_URL),
    );
  } catch {
    return new NextResponse("Administrator access is temporarily unavailable.", {
      status: 503,
      headers: { "Cache-Control": "no-store" },
    });
  }
}

export const config = {
  matcher: ["/admin/:path*"],
};
