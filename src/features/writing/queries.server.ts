import "server-only";

import { unstable_cache } from "next/cache";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import { getStaticWritingOverview } from "./data";
import { mapWritingOverview } from "./mapper";
import { readPublicWritingSource } from "./repository.server";
import type { WritingOverview } from "./types";

const writingTags = [
  publicContentTags.research,
  publicContentTags.writing,
] as const;

const readCachedWritingOverview = unstable_cache(
  async () => mapWritingOverview(await readPublicWritingSource()),
  ["public-writing-overview-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...writingTags],
  },
);

export async function getWritingOverview(): Promise<WritingOverview> {
  const result = await loadPublicContent({
    key: "public-writing-overview-v1",
    tags: writingTags,
    load: readCachedWritingOverview,
    fallback: getStaticWritingOverview,
  });

  return result.data;
}
