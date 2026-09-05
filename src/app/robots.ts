import type { MetadataRoute } from "next";

import { publicEnvironment } from "@/lib/env";
import { privateRobotPaths } from "@/lib/seo-routes";

export default function robots(): MetadataRoute.Robots {
  const origin = new URL(publicEnvironment.NEXT_PUBLIC_SITE_URL).origin;

  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: [...privateRobotPaths],
    },
    sitemap: new URL("/sitemap.xml", origin).toString(),
    host: origin,
  };
}
