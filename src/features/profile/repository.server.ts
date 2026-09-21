import "server-only";

import {
  ContentStatus,
  type Prisma,
} from "@/generated/prisma/client";
import { getDatabase } from "@/lib/db";

import {
  mapPublicProfileRecord,
  mapPublicResumeRecord,
  PUBLIC_RESUME_MIME_TYPES,
} from "./mappers";
import type { ProfilePageData, ResumePageData } from "./types";

const publishedOrder = [
  { sortOrder: "asc" as const },
  { id: "asc" as const },
];

function createProfileSelect(now: Date) {
  return {
    name: true,
    positioning: true,
    headline: true,
    introduction: true,
    about: true,
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
        role: true,
        location: true,
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
      orderBy: publishedOrder,
      select: {
        institution: true,
        degree: true,
        startYear: true,
        startMonth: true,
        endYear: true,
        endMonth: true,
        isCurrent: true,
      },
    },
    researchInterests: {
      where: { status: ContentStatus.PUBLISHED },
      orderBy: publishedOrder,
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
        summary: true,
        format: true,
        stage: true,
        questions: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: { question: true },
        },
      },
    },
    skillCategories: {
      where: { status: ContentStatus.PUBLISHED },
      orderBy: publishedOrder,
      select: {
        name: true,
        skills: {
          where: { status: ContentStatus.PUBLISHED },
          orderBy: publishedOrder,
          select: { name: true },
        },
      },
    },
    certifications: {
      where: { status: ContentStatus.PUBLISHED },
      orderBy: [
        { featured: "desc" },
        { sortOrder: "asc" },
        { id: "asc" },
      ],
      select: {
        name: true,
        issuer: true,
        credentialUrl: true,
        issueYear: true,
        issueMonth: true,
        expiryYear: true,
        expiryMonth: true,
        doesNotExpire: true,
      },
    },
  } satisfies Prisma.ProfileSelect;
}

export async function loadPublicProfilePage(): Promise<ProfilePageData | null> {
  const now = new Date();
  const record = await getDatabase().profile.findFirst({
    where: { key: "primary", status: ContentStatus.PUBLISHED },
    select: createProfileSelect(now),
  });

  return record ? mapPublicProfileRecord(record) : null;
}

export async function loadPublicResumePage(): Promise<ResumePageData | null> {
  const now = new Date();
  const record = await getDatabase().profile.findFirst({
    where: { key: "primary", status: ContentStatus.PUBLISHED },
    select: {
      ...createProfileSelect(now),
      socialLinks: {
        where: {
          status: ContentStatus.PUBLISHED,
          kind: { in: ["EMAIL", "LINKEDIN"] },
        },
        orderBy: publishedOrder,
        select: { kind: true, url: true },
      },
      projects: {
        where: {
          status: ContentStatus.PUBLISHED,
          featured: true,
          publishedAt: { not: null, lte: now },
        },
        orderBy: publishedOrder,
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
            orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
            select: { name: true },
          },
        },
      },
      resumes: {
        where: {
          status: ContentStatus.PUBLISHED,
          isCurrent: true,
          publishedAt: { not: null, lte: now },
          fileAsset: {
            is: {
              publicUrl: { not: null },
              mimeType: { in: [...PUBLIC_RESUME_MIME_TYPES] },
            },
          },
        },
        orderBy: [{ publishedAt: "desc" }, { id: "asc" }],
        take: 1,
        select: {
          fileAsset: {
            select: { publicUrl: true, mimeType: true },
          },
        },
      },
    },
  });

  return record ? mapPublicResumeRecord(record) : null;
}
