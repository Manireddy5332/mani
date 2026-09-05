import "server-only";

import { ContentStatus } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import {
  mapResearchDetail,
  mapResearchOverview,
  type PublicResearchOverviewRow,
  type PublicResearchProjectRow,
} from "./mappers";
import { isPublicResearchSlug } from "./public-slug";
import type { ResearchDetailRecord, ResearchOverview } from "./types";

const PRIMARY_PROFILE_KEY = "primary";

function publicTimedContent(now: Date) {
  return {
    publishedAt: { lte: now },
    status: ContentStatus.PUBLISHED,
  } as const;
}

const questionSelection = {
  id: true,
  question: true,
} as const;

const publicationSelection = {
  arxivUrl: true,
  authors: true,
  id: true,
  paperUrl: true,
  slug: true,
  stage: true,
  title: true,
  venue: true,
  year: true,
} as const;

export async function readResearchOverviewFromDatabase(): Promise<ResearchOverview> {
  const now = new Date();
  const profile = await getDatabase().profile.findFirst({
    where: {
      key: PRIMARY_PROFILE_KEY,
      status: ContentStatus.PUBLISHED,
    },
    select: {
      researchInterests: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: { id: true, name: true },
      },
      researchProjects: {
        where: publicTimedContent(now),
        orderBy: [
          { isCurrent: "desc" },
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        take: 1,
        select: {
          format: true,
          id: true,
          isCurrent: true,
          questions: {
            orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
            select: questionSelection,
          },
          slug: true,
          stage: true,
          summary: true,
          title: true,
        },
      },
      publications: {
        where: publicTimedContent(now),
        orderBy: [
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        select: publicationSelection,
      },
    },
  });

  const row: PublicResearchOverviewRow = profile
    ? {
        currentDirection: profile.researchProjects[0] ?? null,
        interests: profile.researchInterests,
        publications: profile.publications,
      }
    : null;

  return mapResearchOverview(row);
}

export async function readResearchSitemapPathsFromDatabase(): Promise<string[]> {
  const now = new Date();
  const projects = await getDatabase().researchProject.findMany({
    where: {
      ...publicTimedContent(now),
      profile: {
        key: PRIMARY_PROFILE_KEY,
        status: ContentStatus.PUBLISHED,
      },
    },
    orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
    select: { slug: true },
  });

  return projects
    .map(({ slug }) => slug)
    .filter(isPublicResearchSlug)
    .map((slug) => `/research/${slug}`);
}

export async function readResearchDetailFromDatabase(
  slug: string,
): Promise<ResearchDetailRecord | null> {
  if (!isPublicResearchSlug(slug)) return null;

  const now = new Date();
  const research = await getDatabase().researchProject.findFirst({
    where: {
      slug,
      ...publicTimedContent(now),
      profile: {
        key: PRIMARY_PROFILE_KEY,
        status: ContentStatus.PUBLISHED,
      },
    },
    select: {
      evidenceStatus: true,
      format: true,
      id: true,
      isCurrent: true,
      interests: {
        where: { status: ContentStatus.PUBLISHED },
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: { id: true, name: true },
      },
      methodology: true,
      methodologySummary: true,
      projects: {
        where: publicTimedContent(now),
        orderBy: [
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        select: { id: true, shortTitle: true, title: true },
      },
      publications: {
        where: publicTimedContent(now),
        orderBy: [
          { featured: "desc" },
          { sortOrder: "asc" },
          { id: "asc" },
        ],
        select: publicationSelection,
      },
      questions: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: questionSelection,
      },
      scopeBoundary: true,
      slug: true,
      stage: true,
      summary: true,
      technologies: true,
      title: true,
    },
  });

  return research
    ? mapResearchDetail(research satisfies PublicResearchProjectRow)
    : null;
}
