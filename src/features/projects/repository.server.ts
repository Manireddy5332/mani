import "server-only";

import { ContentStatus, Prisma } from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import {
  mapProjectRecord,
  mapProjectsPageContent,
  type PublicProjectRow,
} from "./mappers";
import { isPublicProjectSlug } from "./public-slug";
import type { ProjectDetailRecord, ProjectsPageContent } from "./types";

const PRIMARY_PROFILE_KEY = "primary";

function publicTimedContent(now: Date) {
  return {
    publishedAt: { lte: now },
    status: ContentStatus.PUBLISHED,
  } as const;
}

const projectSelection = {
  advisor: true,
  category: true,
  contributions: {
    orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
    select: { description: true, id: true, label: true },
  },
  description: true,
  features: {
    orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
    select: { id: true, name: true },
  },
  id: true,
  implementation: true,
  institution: true,
  repositoryUrl: true,
  role: true,
  seoDescription: true,
  seoTitle: true,
  shortTitle: true,
  slug: true,
  summary: true,
  technologies: {
    orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
    select: { id: true, name: true },
  },
  title: true,
  type: true,
} satisfies Prisma.ProjectSelect;

export async function readProjectsPageFromDatabase(): Promise<ProjectsPageContent> {
  const now = new Date();
  const projects = await getDatabase().project.findMany({
    where: {
      ...publicTimedContent(now),
      profile: {
        key: PRIMARY_PROFILE_KEY,
        status: ContentStatus.PUBLISHED,
      },
    },
    orderBy: [
      { featured: "desc" },
      { sortOrder: "asc" },
      { id: "asc" },
    ],
    select: projectSelection,
  });

  return mapProjectsPageContent(projects satisfies PublicProjectRow[]);
}

export async function readProjectSitemapPathsFromDatabase(): Promise<string[]> {
  const now = new Date();
  const projects = await getDatabase().project.findMany({
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
    .filter(isPublicProjectSlug)
    .map((slug) => `/projects/${slug}`);
}

export async function readProjectDetailFromDatabase(
  slug: string,
): Promise<ProjectDetailRecord | null> {
  if (!isPublicProjectSlug(slug)) return null;

  const now = new Date();
  const project = await getDatabase().project.findFirst({
    where: {
      slug,
      ...publicTimedContent(now),
      profile: {
        key: PRIMARY_PROFILE_KEY,
        status: ContentStatus.PUBLISHED,
      },
    },
    select: projectSelection,
  });

  return project ? mapProjectRecord(project satisfies PublicProjectRow) : null;
}
