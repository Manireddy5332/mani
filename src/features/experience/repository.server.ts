import "server-only";

import { ContentStatus } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import type { ExperiencePageSource } from "./types";

export async function readPublicExperienceSource(): Promise<ExperiencePageSource> {
  const profile = await getDatabase().profile.findFirst({
    where: {
      key: "primary",
      status: ContentStatus.PUBLISHED,
    },
    select: {
      experiences: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          type: true,
          location: true,
          organization: true,
          practiceAreas: true,
          role: true,
          summary: true,
          startYear: true,
          startMonth: true,
          endYear: true,
          endMonth: true,
          isCurrent: true,
        },
      },
      educationRecords: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          degree: true,
          institution: true,
          startYear: true,
          startMonth: true,
          endYear: true,
          endMonth: true,
          isCurrent: true,
        },
      },
      skillCategories: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          name: true,
          skills: {
            where: { status: ContentStatus.PUBLISHED },
            orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
            select: { name: true },
          },
        },
      },
    },
  });

  return {
    experiences: profile?.experiences ?? [],
    education: profile?.educationRecords ?? [],
    skillCategories: profile?.skillCategories ?? [],
  };
}
