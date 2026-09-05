import "server-only";

import { unstable_cache } from "next/cache";
import { cache } from "react";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import {
  getResearchDetailBySlug as getStaticResearchDetailBySlug,
  getResearchOverview as getStaticResearchOverview,
} from "./data";
import {
  readResearchDetailFromDatabase,
  readResearchOverviewFromDatabase,
  readResearchSitemapPathsFromDatabase,
} from "./repository.server";
import { isPublicResearchSlug } from "./public-slug";
import type { ResearchDetailRecord, ResearchOverview } from "./types";

const researchTags = [
  publicContentTags.profile,
  publicContentTags.research,
  publicContentTags.publications,
  publicContentTags.projects,
] as const;

const readCachedResearchOverview = unstable_cache(
  readResearchOverviewFromDatabase,
  ["public-research-overview"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...researchTags],
  },
);

const readCachedResearchDetail = unstable_cache(
  readResearchDetailFromDatabase,
  ["public-research-detail"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...researchTags],
  },
);

const readCachedResearchSitemapPaths = unstable_cache(
  readResearchSitemapPathsFromDatabase,
  ["public-research-sitemap-paths"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...researchTags],
  },
);

export async function getResearchOverview(): Promise<ResearchOverview> {
  const result = await loadPublicContent({
    key: "research:overview",
    tags: researchTags,
    load: readCachedResearchOverview,
    fallback: getStaticResearchOverview,
  });
  return result.data;
}

async function loadResearchDetailBySlug(
  slug: string,
): Promise<ResearchDetailRecord | null> {
  if (!isPublicResearchSlug(slug)) return null;

  const result = await loadPublicContent<ResearchDetailRecord | null>({
    key: `research:detail:${slug}`,
    tags: researchTags,
    load: () => readCachedResearchDetail(slug),
    fallback: () => getStaticResearchDetailBySlug(slug) ?? null,
  });
  return result.data;
}

export const getResearchDetailBySlug = cache(loadResearchDetailBySlug);

export async function getPublicResearchSitemapPaths(): Promise<string[]> {
  try {
    return await readCachedResearchSitemapPaths();
  } catch {
    return [];
  }
}
