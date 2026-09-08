export type PublicStatus = "DRAFT" | "PUBLISHED" | "ARCHIVED";

export function isPubliclyEligible(
  record: Readonly<{
    status: PublicStatus;
    publishedAt: Date | string | null;
  }>,
  now: Date,
): boolean {
  if (record.status !== "PUBLISHED" || record.publishedAt === null) {
    return false;
  }

  const publishedAt =
    record.publishedAt instanceof Date
      ? record.publishedAt
      : new Date(record.publishedAt);
  return Number.isFinite(publishedAt.getTime()) && publishedAt <= now;
}

