import { z } from "zod";

import { isSecureApplicationOrigin } from "@/lib/origin-policy";

const publicEnvironmentSchema = z.object({
  NEXT_PUBLIC_SITE_URL: z
    .url("NEXT_PUBLIC_SITE_URL must be an absolute URL.")
    .refine(
      isSecureApplicationOrigin,
      "NEXT_PUBLIC_SITE_URL must be an HTTPS origin, or an HTTP loopback origin for local development.",
    ),
});

export const publicEnvironment = publicEnvironmentSchema.parse({
  NEXT_PUBLIC_SITE_URL:
    process.env.NEXT_PUBLIC_SITE_URL ??
    (process.env.NODE_ENV === "production"
      ? undefined
      : "http://localhost:3000"),
});
