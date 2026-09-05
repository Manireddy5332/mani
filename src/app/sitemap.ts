import type { MetadataRoute } from "next";

import { getPublicProjectSitemapPaths } from "@/features/projects";
import { getPublicResearchSitemapPaths } from "@/features/research";
import { publicEnvironment } from "@/lib/env";
import { buildPublicSitemapPaths } from "@/lib/seo-routes";

export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const [projectPaths, researchPaths] = await Promise.all([
    getPublicProjectSitemapPaths(),
    getPublicResearchSitemapPaths(),
  ]);
  const origin = new URL(publicEnvironment.NEXT_PUBLIC_SITE_URL).origin;
  const paths = buildPublicSitemapPaths(projectPaths, researchPaths);

  return paths.map((path) => ({
    url: new URL(path, origin).toString(),
  }));
}
