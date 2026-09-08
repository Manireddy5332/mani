import type { Metadata } from "next";

import { siteConfig } from "@/lib/site";

const PAGE_TITLE_LIMIT = 70;
const PAGE_DESCRIPTION_LIMIT = 160;

export type PublicPageMetadataInput = {
  readonly description: string;
  readonly path: `/${string}`;
  readonly title: string;
};

export function normalizeMetadataText(value: string, limit: number): string {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, Math.max(1, limit - 1)).trimEnd()}…`;
}

/** Consistent, bounded metadata for public index and detail routes. */
export function createPublicPageMetadata({
  description,
  path,
  title,
}: PublicPageMetadataInput): Metadata {
  const pageTitle = normalizeMetadataText(title, PAGE_TITLE_LIMIT);
  const pageDescription = normalizeMetadataText(
    description,
    PAGE_DESCRIPTION_LIMIT,
  );
  const socialTitle = `${pageTitle} · ${siteConfig.name}`;

  return {
    title: pageTitle,
    description: pageDescription,
    alternates: { canonical: path },
    openGraph: {
      type: "website",
      title: socialTitle,
      description: pageDescription,
      url: path,
      siteName: `${siteConfig.name} Portfolio`,
    },
    twitter: {
      card: "summary",
      title: socialTitle,
      description: pageDescription,
    },
  };
}
