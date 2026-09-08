export type AdminIdentity = {
  email?: string | null;
  emailVerified?: boolean | null;
};

export type SessionLike = {
  user?: AdminIdentity | null;
} | null;

export type AdminAccessDecision =
  | "authorized"
  | "forbidden"
  | "unauthenticated";

/**
 * Email comparison deliberately performs only whitespace and case
 * normalization. Provider-specific aliases (such as dots or plus-addressing)
 * must never expand the configured administrator allowlist.
 */
export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function isAdminIdentity(
  identity: AdminIdentity | null | undefined,
  configuredAdminEmail: string,
): boolean {
  if (
    identity?.emailVerified !== true ||
    typeof identity.email !== "string" ||
    configuredAdminEmail.trim() === ""
  ) {
    return false;
  }

  return normalizeEmail(identity.email) === normalizeEmail(configuredAdminEmail);
}

export function decideAdminAccess(
  session: SessionLike,
  configuredAdminEmail: string,
): AdminAccessDecision {
  if (!session?.user) {
    return "unauthenticated";
  }

  return isAdminIdentity(session.user, configuredAdminEmail)
    ? "authorized"
    : "forbidden";
}
