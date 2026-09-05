import "server-only";

import { unstable_cache } from "next/cache";
import { cache } from "react";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import {
  getProjectDetailBySlug as getStaticProjectDetailBySlug,
  getProjectsPageContent as getStaticProjectsPageContent,
} from "./content";
import {
  readProjectDetailFromDatabase,
  readProjectSitemapPathsFromDatabase,
  readProjectsPageFromDatabase,
} from "./repository.server";
import { isPublicProjectSlug } from "./public-slug";
import type { ProjectDetailRecord, ProjectsPageContent } from "./types";

const projectTags = [
  publicContentTags.projects,
  publicContentTags.profile,
] as const;

const readCachedProjectsPage = unstable_cache(
  readProjectsPageFromDatabase,
  ["public-projects-overview"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...projectTags],
  },
);

const readCachedProjectDetail = unstable_cache(
  readProjectDetailFromDatabase,
  ["public-project-detail"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...projectTags],
  },
);

const readCachedProjectSitemapPaths = unstable_cache(
  readProjectSitemapPathsFromDatabase,
  ["public-project-sitemap-paths"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...projectTags],
  },
);

export async function getProjectsPageContent(): Promise<ProjectsPageContent> {
  const result = await loadPublicContent({
    key: "projects:overview",
    tags: projectTags,
    load: readCachedProjectsPage,
    fallback: getStaticProjectsPageContent,
  });
  return result.data;
}

async function loadProjectDetailBySlug(
  slug: string,
): Promise<ProjectDetailRecord | null> {
  if (!isPublicProjectSlug(slug)) return null;

  const result = await loadPublicContent<ProjectDetailRecord | null>({
    key: `projects:detail:${slug}`,
    tags: projectTags,
    load: () => readCachedProjectDetail(slug),
    fallback: () => getStaticProjectDetailBySlug(slug) ?? null,
  });
  return result.data;
}

export const getProjectDetailBySlug = cache(loadProjectDetailBySlug);

export async function getPublicProjectSitemapPaths(): Promise<string[]> {
  try {
    return await readCachedProjectSitemapPaths();
  } catch {
    return [];
  }
}
