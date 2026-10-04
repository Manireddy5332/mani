import type { BeforeSendEvent } from "@vercel/analytics";

import { publicIndexPaths } from "./seo-routes";

const publicPaths: ReadonlySet<string> = new Set(publicIndexPaths);
const publicDetailPath = /^\/(research|projects)\/[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** Vercel's supported event filter also runs after client-side navigation. */
export function beforeSendPublicPageview(
  event: BeforeSendEvent,
): BeforeSendEvent | null {
  // Phase 1 collects page views only, never custom events or their properties.
  if (event.type !== "pageview") return null;

  try {
    const url = new URL(event.url);
    const pathname = url.pathname.replace(/\/+$/, "") || "/";

    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      (!publicPaths.has(pathname) && !publicDetailPath.test(pathname))
    ) {
      return null;
    }

    // Queries/fragments can contain OAuth codes, email addresses or other PII.
    // Keep only the public page URL; unknown/private routes fail closed.
    return { type: "pageview", url: `${url.origin}${pathname}` };
  } catch {
    return null;
  }
}
