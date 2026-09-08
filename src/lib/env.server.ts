import "server-only";

import { z } from "zod";

import {
  applicationOriginsMatch,
  isSecureApplicationOrigin,
} from "@/lib/origin-policy";

const postgresUrl = z
  .string()
  .min(1, "DATABASE_URL is required when database access is used.")
  .refine(
    (value) => value.startsWith("postgresql://") || value.startsWith("postgres://"),
    "DATABASE_URL must be a PostgreSQL connection string.",
  );

const authBaseUrl = z
  .url("BETTER_AUTH_URL must be an absolute URL.")
  .refine(
    isSecureApplicationOrigin,
    "BETTER_AUTH_URL must be an HTTPS origin, or an HTTP loopback origin for local development.",
  );

const normalizedEmail = z
  .string()
  .trim()
  .toLowerCase()
  .pipe(z.email("ADMIN_EMAIL must be a valid email address."));

const serverEnvironmentSchema = z
  .object({
    DATABASE_URL: postgresUrl,
    BETTER_AUTH_SECRET: z
      .string()
      .min(32, "BETTER_AUTH_SECRET must contain at least 32 characters."),
    BETTER_AUTH_URL: authBaseUrl,
    GOOGLE_CLIENT_ID: z.string().min(1, "GOOGLE_CLIENT_ID is required."),
    GOOGLE_CLIENT_SECRET: z
      .string()
      .min(1, "GOOGLE_CLIENT_SECRET is required."),
    ADMIN_EMAIL: normalizedEmail,
    NEXT_PUBLIC_SITE_URL: authBaseUrl,
  })
  .refine(
    ({ BETTER_AUTH_URL, NEXT_PUBLIC_SITE_URL }) =>
      applicationOriginsMatch(BETTER_AUTH_URL, NEXT_PUBLIC_SITE_URL),
    {
      path: ["BETTER_AUTH_URL"],
      message:
        "BETTER_AUTH_URL and NEXT_PUBLIC_SITE_URL must use the same origin.",
    },
  );

export type ServerEnvironment = z.infer<typeof serverEnvironmentSchema>;

let cachedServerEnvironment: ServerEnvironment | undefined;

/**
 * Parses private runtime configuration only when a server-only database or
 * authentication boundary asks for it. Secret values are never exported to a
 * client module.
 */
export function getServerEnvironment(): ServerEnvironment {
  cachedServerEnvironment ??= serverEnvironmentSchema.parse({
    DATABASE_URL: process.env.DATABASE_URL,
    BETTER_AUTH_SECRET: process.env.BETTER_AUTH_SECRET,
    BETTER_AUTH_URL: process.env.BETTER_AUTH_URL,
    GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
    ADMIN_EMAIL: process.env.ADMIN_EMAIL,
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL,
  });

  return cachedServerEnvironment;
}
