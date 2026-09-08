import "server-only";

import { unstable_cache } from "next/cache";
import { cache } from "react";

import {
  PUBLIC_CONTENT_CACHE_SECONDS,
  publicContentTags,
} from "@/lib/public-content/cache-policy";
import { loadPublicContent } from "@/lib/public-content/fallback.server";

import { getProfilePageData, getResumePageData } from "./data";
import {
  loadPublicProfilePage,
  loadPublicResumePage,
} from "./repository.server";
import type { ProfilePageData, ResumePageData } from "./types";

const profileTags = [
  publicContentTags.profile,
  publicContentTags.experience,
  publicContentTags.education,
  publicContentTags.research,
  publicContentTags.skills,
  publicContentTags.certifications,
] as const;

const resumeTags = [
  ...profileTags,
  publicContentTags.socialLinks,
  publicContentTags.projects,
  publicContentTags.resumes,
] as const;

const loadCachedProfilePage = unstable_cache(
  loadPublicProfilePage,
  ["public-profile-page-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...profileTags],
  },
);

const loadCachedResumePage = unstable_cache(
  loadPublicResumePage,
  ["public-resume-page-v1"],
  {
    revalidate: PUBLIC_CONTENT_CACHE_SECONDS,
    tags: [...resumeTags],
  },
);

export const getPublicProfilePageData = cache(
  async (): Promise<ProfilePageData | null> => {
    const result = await loadPublicContent<ProfilePageData | null>({
      key: "public-profile-page-v1",
      tags: profileTags,
      load: loadCachedProfilePage,
      fallback: getProfilePageData,
    });
    return result.data;
  },
);

export const getPublicResumePageData = cache(
  async (): Promise<ResumePageData | null> => {
    const result = await loadPublicContent<ResumePageData | null>({
      key: "public-resume-page-v1",
      tags: resumeTags,
      load: loadCachedResumePage,
      fallback: getResumePageData,
    });
    return result.data;
  },
);
