import "server-only";

import { unstable_cache } from "next/cache";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import { getStaticExperiencePageContent } from "./content";
import { mapExperiencePageContent } from "./mapper";
import { readPublicExperienceSource } from "./repository.server";
import type { ExperiencePageContent } from "./types";

const experienceTags = [
  publicContentTags.experience,
  publicContentTags.education,
  publicContentTags.skills,
] as const;

const readCachedExperiencePageContent = unstable_cache(
  async () => mapExperiencePageContent(await readPublicExperienceSource()),
  ["public-experience-page-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...experienceTags],
  },
);

export async function getExperiencePageContent(): Promise<ExperiencePageContent> {
  const result = await loadPublicContent({
    key: "public-experience-page-v1",
    tags: experienceTags,
    load: readCachedExperiencePageContent,
    fallback: getStaticExperiencePageContent,
  });

  return result.data;
}
