"use client";

import { createAuthClient } from "better-auth/react";

/**
 * The browser uses same-origin Better Auth routes. No server environment or
 * OAuth credential is included in this client module.
 */
export const authClient = createAuthClient();
