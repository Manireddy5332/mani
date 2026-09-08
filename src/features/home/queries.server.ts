import "server-only";

import { unstable_cache } from "next/cache";
import { cache } from "react";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import { getStaticHomePageData } from "./data";
import { readHomePageFromDatabase } from "./repository.server";
import type { HomePageData } from "./types";

const homeTags = [
  publicContentTags.profile,
  publicContentTags.socialLinks,
  publicContentTags.experience,
  publicContentTags.education,
  publicContentTags.research,
  publicContentTags.projects,
  publicContentTags.skills,
] as const;

const readCachedHomePage = unstable_cache(
  readHomePageFromDatabase,
  ["public-home-page-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...homeTags],
  },
);

export const getPublicHomePageData = cache(
  async (): Promise<HomePageData | null> => {
    const result = await loadPublicContent<HomePageData | null>({
      key: "public-home-page-v1",
      tags: homeTags,
      load: readCachedHomePage,
      fallback: getStaticHomePageData,
    });
    return result.data;
  },
);

