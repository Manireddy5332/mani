import "server-only";

import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { cache } from "react";

import { getServerEnvironment } from "@/lib/env.server";

import { decideAdminAccess } from "./policy";
import { getAuth, type AuthInstance } from "./server";

export type AdminSession = NonNullable<
  Awaited<ReturnType<AuthInstance["api"]["getSession"]>>
>;

export type AdminAccess =
  | { status: "authorized"; session: AdminSession }
  | { status: "forbidden" }
  | { status: "unauthenticated" }
  | { status: "unavailable" };

export type AdminAccessErrorCode =
  | "FORBIDDEN"
  | "UNAUTHENTICATED"
  | "UNAVAILABLE";

export class AdminAccessError extends Error {
  readonly code: AdminAccessErrorCode;

  constructor(code: AdminAccessErrorCode) {
    super("Administrator authorization failed.");
    this.name = "AdminAccessError";
    this.code = code;
  }
}

async function resolveAdminAccess(requestHeaders: Headers): Promise<AdminAccess> {
  try {
    const session = await getAuth().api.getSession({
      headers: requestHeaders,
    });
    const decision = decideAdminAccess(
      session,
      getServerEnvironment().ADMIN_EMAIL,
    );

    if (decision === "authorized" && session) {
      return { status: "authorized", session };
    }

    if (decision === "forbidden") {
      return { status: "forbidden" };
    }

    return { status: "unauthenticated" };
  } catch {
    return { status: "unavailable" };
  }
}

const getRequestAdminAccess = cache(async () =>
  resolveAdminAccess(await headers()),
);

export async function getAdminAccess(
  requestHeaders?: Headers,
): Promise<AdminAccess> {
  return requestHeaders
    ? resolveAdminAccess(requestHeaders)
    : getRequestAdminAccess();
}

/**
 * Server action/query guard. It never redirects and never returns an
 * unauthorized session, which makes it safe to place before all validation
 * and database work.
 */
export async function assertAdmin(
  requestHeaders?: Headers,
): Promise<AdminSession> {
  const access = await getAdminAccess(requestHeaders);

  if (access.status === "authorized") {
    return access.session;
  }

  if (access.status === "unauthenticated") {
    throw new AdminAccessError("UNAUTHENTICATED");
  }

  if (access.status === "forbidden") {
    throw new AdminAccessError("FORBIDDEN");
  }

  throw new AdminAccessError("UNAVAILABLE");
}

/**
 * Page/layout guard. Authorization is complete before either redirect or the
 * protected component tree is rendered.
 */
export async function requireAdmin(
  requestHeaders?: Headers,
): Promise<AdminSession> {
  const access = await getAdminAccess(requestHeaders);

  if (access.status === "authorized") {
    return access.session;
  }

  if (access.status === "unauthenticated") {
    redirect("/sign-in");
  }

  redirect("/access-denied");
}
