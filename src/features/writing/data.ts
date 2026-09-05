import { portfolioContent } from "@/lib/portfolio";

import type { WritingOverview } from "./types";

const writingOverview = {
  sourceDirection: {
    title: portfolioContent.researchInterests[0],
    ...portfolioContent.currentResearch,
  },
  entries: [],
} as const satisfies WritingOverview;

/**
 * Verified static fallback for database-read failures. A successful empty
 * database result remains empty, and no article is exposed in Phase 7.
 */
export function getStaticWritingOverview(): WritingOverview {
  return writingOverview;
}
