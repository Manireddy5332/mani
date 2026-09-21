import "server-only";

import { ContentStatus, type Prisma } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import { mapPublicHomeRecord } from "./mappers";
import type { HomePageData } from "./types";

const ordered = [
  { sortOrder: "asc" as const },
  { id: "asc" as const },
];

export async function readHomePageFromDatabase(): Promise<HomePageData | null> {
  const now = new Date();
  const profile = await getDatabase().profile.findFirst({
    where: { key: "primary", status: ContentStatus.PUBLISHED },
    select: {
      name: true,
      positioning: true,
      introduction: true,
      location: true,
      avatar: {
        select: {
          id: true,
          provider: true,
          mimeType: true,
          width: true,
          height: true,
        },
      },
      socialLinks: {
        where: {
          status: ContentStatus.PUBLISHED,
          kind: { in: ["EMAIL", "LINKEDIN"] },
        },
        orderBy: ordered,
        select: { kind: true, url: true },
      },
      experiences: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [
          { isCurrent: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        select: {
          type: true,
          organization: true,
          location: true,
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
        orderBy: ordered,
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
      researchInterests: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: ordered,
        select: { name: true },
      },
      researchProjects: {
        where: {
          status: ContentStatus.PUBLISHED,
          publishedAt: { not: null, lte: now },
        },
        orderBy: [
          { isCurrent: "desc" },
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        take: 1,
        select: {
          slug: true,
          title: true,
          stage: true,
          format: true,
          summary: true,
          questions: {
            orderBy: ordered,
            select: { question: true },
          },
        },
      },
      projects: {
        where: {
          status: ContentStatus.PUBLISHED,
          publishedAt: { not: null, lte: now },
        },
        orderBy: [
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        take: 1,
        select: {
          slug: true,
          title: true,
          type: true,
          institution: true,
          advisor: true,
          role: true,
          summary: true,
          repositoryUrl: true,
          technologies: {
            orderBy: ordered,
            select: { name: true },
          },
        },
      },
      skillCategories: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: ordered,
        select: {
          name: true,
          skills: {
            where: { status: ContentStatus.PUBLISHED },
            orderBy: ordered,
            select: { name: true },
          },
        },
      },
    } satisfies Prisma.ProfileSelect,
  });

  return profile ? mapPublicHomeRecord(profile) : null;
}
