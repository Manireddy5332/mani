import "server-only";

import { ContentStatus } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import type { WritingOverviewSource } from "./types";

export async function readPublicWritingSource(
  now = new Date(),
): Promise<WritingOverviewSource> {
  const sourceDirection = await getDatabase().researchProject.findFirst({
    where: {
      isCurrent: true,
      status: ContentStatus.PUBLISHED,
      publishedAt: {
        not: null,
        lte: now,
      },
      profile: {
        key: "primary",
        status: ContentStatus.PUBLISHED,
      },
    },
    orderBy: [
      { featured: "desc" },
      { sortOrder: "asc" },
      { publishedAt: "desc" },
      { id: "asc" },
    ],
    select: {
      title: true,
      stage: true,
      format: true,
      summary: true,
      questions: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: { question: true },
      },
    },
  });

  return { sourceDirection };
}
