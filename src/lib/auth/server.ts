import "server-only";

import { prismaAdapter } from "@better-auth/prisma-adapter";
import { betterAuth } from "better-auth";
import { APIError } from "better-auth/api";

import { getDatabase } from "@/lib/db";
import { getServerEnvironment } from "@/lib/env.server";
import { siteConfig } from "@/lib/site";

import { isAdminIdentity } from "./policy";

const SESSION_LENGTH_SECONDS = 60 * 60 * 24 * 7;
const SESSION_REFRESH_SECONDS = 60 * 60 * 24;

function denyAuthentication(): never {
  throw new APIError("FORBIDDEN", {
    message: "Access denied.",
  });
}

function createAuthInstance() {
  const environment = getServerEnvironment();
  const trustedOrigin = new URL(environment.BETTER_AUTH_URL).origin;

  return betterAuth({
    appName: siteConfig.brandName,
    baseURL: environment.BETTER_AUTH_URL,
    secret: environment.BETTER_AUTH_SECRET,
    trustedOrigins: [trustedOrigin],
    database: prismaAdapter(getDatabase(), {
      provider: "postgresql",
      transaction: true,
    }),
    socialProviders: {
      google: {
        clientId: environment.GOOGLE_CLIENT_ID,
        clientSecret: environment.GOOGLE_CLIENT_SECRET,
        overrideUserInfoOnSignIn: true,
        prompt: "select_account",
      },
    },
    account: {
      encryptOAuthTokens: true,
      storeStateStrategy: "database",
      accountLinking: {
        enabled: false,
        disableImplicitLinking: true,
        allowDifferentEmails: false,
        allowUnlinkingAll: false,
      },
    },
    session: {
      expiresIn: SESSION_LENGTH_SECONDS,
      updateAge: SESSION_REFRESH_SECONDS,
      cookieCache: {
        enabled: false,
      },
    },
    databaseHooks: {
      account: {
        create: {
          before: async (account) => ({
            // Better Auth 1.6 encrypts access and refresh tokens, but its
            // Google flow can still persist the ID token verbatim. The
            // portfolio does not use stored ID tokens, so discard it.
            data: { ...account, idToken: null },
          }),
        },
        update: {
          before: async (account) => ({
            data: { ...account, idToken: null },
          }),
        },
      },
      user: {
        create: {
          before: async (user) => {
            if (!isAdminIdentity(user, environment.ADMIN_EMAIL)) {
              denyAuthentication();
            }

            return { data: user };
          },
        },
      },
      session: {
        create: {
          before: async (session, context) => {
            const user = context
              ? await context.context.internalAdapter.findUserById(session.userId)
              : null;

            if (!isAdminIdentity(user, environment.ADMIN_EMAIL)) {
              denyAuthentication();
            }

            return { data: session };
          },
        },
      },
    },
    rateLimit: {
      enabled: true,
      window: 60,
      max: 100,
      storage: "memory",
    },
    telemetry: {
      enabled: false,
    },
  });
}

export type AuthInstance = ReturnType<typeof createAuthInstance>;

const globalForAuth = globalThis as typeof globalThis & {
  portfolioAuth?: AuthInstance;
};

/**
 * Better Auth is initialized on first server use so static public routes and
 * credential-free tooling do not eagerly open a database-backed auth runtime.
 */
export function getAuth(): AuthInstance {
  globalForAuth.portfolioAuth ??= createAuthInstance();

  return globalForAuth.portfolioAuth;
}
