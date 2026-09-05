type AuthMutationResult = {
  error?: unknown;
};

type SecureSignOutClient = {
  revokeSessions: () => Promise<AuthMutationResult>;
  signOut: () => Promise<AuthMutationResult>;
};

/**
 * Better Auth's regular sign-out route intentionally clears the cookie even
 * when its database deletion fails. Revoke the single-owner admin sessions
 * through the error-reporting endpoint first so the UI cannot report success
 * while a server-side session remains active.
 */
export async function performSecureSignOut(
  client: SecureSignOutClient,
): Promise<{ ok: boolean }> {
  const revokeResult = await client.revokeSessions();
  if (revokeResult.error) return { ok: false };

  const signOutResult = await client.signOut();
  return { ok: !signOutResult.error };
}
