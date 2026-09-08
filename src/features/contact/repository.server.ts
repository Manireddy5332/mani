import "server-only";

import { ContentStatus, type Prisma } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import { mapPublicContactRecord } from "./mappers";
import type { ContactPageData } from "./types";

export async function readContactPageFromDatabase(): Promise<ContactPageData | null> {
  const now = new Date();
  const profile = await getDatabase().profile.findFirst({
    where: { key: "primary", status: ContentStatus.PUBLISHED },
    select: {
      name: true,
      location: true,
      socialLinks: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        select: {
          key: true,
          kind: true,
          label: true,
          url: true,
          handle: true,
        },
      },
      projects: {
        where: {
          status: ContentStatus.PUBLISHED,
          publishedAt: { not: null, lte: now },
          repositoryUrl: { not: null },
        },
        orderBy: [
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        take: 1,
        select: { title: true, shortTitle: true, repositoryUrl: true },
      },
    } satisfies Prisma.ProfileSelect,
  });

  return profile ? mapPublicContactRecord(profile) : null;
}

