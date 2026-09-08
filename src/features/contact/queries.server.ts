import "server-only";

import { unstable_cache } from "next/cache";
import { cache } from "react";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import { getStaticContactPageData } from "./data";
import { readContactPageFromDatabase } from "./repository.server";
import type { ContactPageData } from "./types";

const contactTags = [
  publicContentTags.profile,
  publicContentTags.socialLinks,
  publicContentTags.projects,
] as const;

const readCachedContactPage = unstable_cache(
  readContactPageFromDatabase,
  ["public-contact-page-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...contactTags],
  },
);

export const getPublicContactPageData = cache(
  async (): Promise<ContactPageData | null> => {
    const result = await loadPublicContent<ContactPageData | null>({
      key: "public-contact-page-v1",
      tags: contactTags,
      load: readCachedContactPage,
      fallback: getStaticContactPageData,
    });
    return result.data;
  },
);

