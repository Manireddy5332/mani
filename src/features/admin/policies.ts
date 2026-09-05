import type {
  AdminResourceKey,
  AdminVisibilityStatus,
} from "@/features/admin/types";
import {
  publicContentTagsByAdminResource,
  type PublicContentTag,
} from "@/lib/public-content/cache-policy";

export type AdminPublicationStatus =
  | "ARCHIVED"
  | AdminVisibilityStatus;

/**
 * Applies the portfolio publication lifecycle without changing a retained
 * schedule. A new publication receives `now`; draft/archive transitions clear
 * the timestamp so private content cannot accidentally pass public queries.
 */
export function resolvePublishedAt(
  status: AdminPublicationStatus,
  current: Date | null,
  now = new Date(),
): Date | null {
  return status === "PUBLISHED" ? (current ?? now) : null;
}

export function isPublicationStateValid(
  status: AdminPublicationStatus,
  publishedAt: Date | null,
): boolean {
  return status === "PUBLISHED"
    ? publishedAt !== null
    : publishedAt === null;
}

export function withCanonicalSortOrder<T>(
  items: readonly T[],
): Array<T & { sortOrder: number }> {
  return items.map((item, sortOrder) => ({ ...item, sortOrder }));
}

export type AdminRemovalDecision =
  | "confirmation-required"
  | "mismatch"
  | "proceed";

export function decideAdminRemoval(
  actualIds: readonly string[],
  suppliedIds: readonly string[],
  confirmed: boolean,
): AdminRemovalDecision {
  const actual = new Set(actualIds);
  const supplied = new Set(suppliedIds);
  if (
    actualIds.length !== suppliedIds.length ||
    actual.size !== supplied.size ||
    [...actual].some((id) => !supplied.has(id))
  ) {
    return "mismatch";
  }
  return actual.size > 0 && !confirmed ? "confirmation-required" : "proceed";
}

export function revisionMatches(
  suppliedRevision: string,
  current: Date,
): boolean {
  const suppliedTimestamp = Date.parse(suppliedRevision);
  return (
    Number.isFinite(suppliedTimestamp) && suppliedTimestamp === current.getTime()
  );
}

const publicPathsByAdminResource = {
  profile: ["/", "/about", "/resume", "/contact"],
  "social-links": ["/", "/about", "/resume", "/contact"],
  "research-interests": ["/", "/research", "/about", "/resume"],
  "research-projects": ["/", "/research", "/about", "/resume", "/writing"],
  projects: ["/", "/projects", "/resume", "/contact"],
  experience: ["/", "/experience", "/about", "/resume"],
  education: ["/", "/experience", "/about", "/resume"],
  publications: ["/", "/research"],
  articles: ["/writing"],
  resumes: ["/resume"],
  "skill-categories": ["/", "/experience", "/about", "/resume"],
  skills: ["/", "/experience", "/about", "/resume"],
  certifications: ["/about", "/resume"],
  "site-settings": [],
} as const satisfies Record<AdminResourceKey, readonly string[]>;

export type AdminInvalidationPlan = {
  tags: readonly PublicContentTag[];
  publicPaths: readonly string[];
  adminPaths: readonly string[];
};

export function getAdminInvalidationPlan(
  resource: AdminResourceKey,
  id?: string,
): AdminInvalidationPlan {
  return {
    tags: publicContentTagsByAdminResource[resource],
    publicPaths: publicPathsByAdminResource[resource],
    adminPaths: [
      `/admin/${resource}`,
      ...(id ? [`/admin/${resource}/${id}/edit`] : []),
    ],
  };
}
