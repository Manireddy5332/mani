import "server-only";

import {
  ContentSource,
  ContentStatus,
} from "@/generated/prisma/client";
import type {
  AdminCreateInputMap,
  AdminUpdateInputMap,
} from "@/features/admin/schemas";
import {
  decideAdminRemoval,
  isPublicationStateValid,
  revisionMatches,
  resolvePublishedAt,
  withCanonicalSortOrder,
} from "@/features/admin/policies";
import {
  adminResourceKeys,
  type AdminDashboardDto,
  type AdminFieldValue,
  type AdminNestedContentDto,
  type AdminNestedContentInputMap,
  type AdminNestedParentDto,
  type AdminNestedRemovalConfirmation,
  type AdminNestedResourceKey,
  type AdminProjectNestedContentDto,
  type AdminRecordDto,
  type AdminRecordSummaryDto,
  type AdminResearchNestedContentDto,
  type AdminResourceKey,
  type AdminVisibilityStatus,
} from "@/features/admin/types";
import { getDatabase } from "@/lib/db";

const PRIMARY_PROFILE_KEY = "primary";

const editableFields = {
  profile: [
    "slug",
    "name",
    "positioning",
    "headline",
    "introduction",
    "about",
    "location",
    "status",
  ],
  "social-links": [
    "key",
    "kind",
    "label",
    "url",
    "handle",
    "sortOrder",
    "status",
  ],
  "research-interests": [
    "slug",
    "name",
    "description",
    "sortOrder",
    "status",
  ],
  "research-projects": [
    "slug",
    "title",
    "summary",
    "abstract",
    "motivation",
    "methodology",
    "methodologySummary",
    "domain",
    "researchArea",
    "format",
    "scopeBoundary",
    "evidenceStatus",
    "advisor",
    "collaborators",
    "technologies",
    "datasets",
    "githubUrl",
    "paperUrl",
    "progressPercent",
    "stage",
    "startYear",
    "startMonth",
    "endYear",
    "endMonth",
    "isCurrent",
    "featured",
    "status",
    "sortOrder",
  ],
  projects: [
    "slug",
    "title",
    "shortTitle",
    "type",
    "category",
    "summary",
    "description",
    "problemStatement",
    "motivation",
    "approach",
    "architecture",
    "aiModels",
    "datasets",
    "challenges",
    "results",
    "lessons",
    "institution",
    "advisor",
    "role",
    "implementation",
    "repositoryUrl",
    "demoUrl",
    "documentationUrl",
    "seoTitle",
    "seoDescription",
    "projectStatus",
    "startYear",
    "startMonth",
    "endYear",
    "endMonth",
    "isCurrent",
    "featured",
    "status",
    "sortOrder",
  ],
  experience: [
    "slug",
    "type",
    "organization",
    "role",
    "location",
    "summary",
    "domain",
    "practiceAreas",
    "highlights",
    "startYear",
    "startMonth",
    "endYear",
    "endMonth",
    "isCurrent",
    "featured",
    "status",
    "sortOrder",
  ],
  education: [
    "slug",
    "institution",
    "degree",
    "field",
    "location",
    "startYear",
    "startMonth",
    "endYear",
    "endMonth",
    "isCurrent",
    "gpa",
    "gpaScale",
    "coursework",
    "thesis",
    "activities",
    "url",
    "status",
    "sortOrder",
  ],
  publications: [
    "slug",
    "title",
    "authors",
    "venue",
    "type",
    "year",
    "abstract",
    "doi",
    "arxivUrl",
    "paperUrl",
    "citation",
    "bibtex",
    "stage",
    "featured",
    "status",
    "sortOrder",
  ],
  articles: [
    "slug",
    "title",
    "excerpt",
    "content",
    "contentFormat",
    "category",
    "estimatedReadingMinutes",
    "seoTitle",
    "seoDescription",
    "featured",
    "status",
    "sortOrder",
  ],
  resumes: ["slug", "title", "version", "isCurrent", "status"],
  "skill-categories": [
    "slug",
    "name",
    "description",
    "sortOrder",
    "status",
  ],
  skills: [
    "categoryId",
    "slug",
    "name",
    "description",
    "sortOrder",
    "status",
  ],
  certifications: [
    "slug",
    "name",
    "issuer",
    "credentialId",
    "credentialUrl",
    "description",
    "issueYear",
    "issueMonth",
    "expiryYear",
    "expiryMonth",
    "doesNotExpire",
    "featured",
    "status",
    "sortOrder",
  ],
  "site-settings": [],
} as const satisfies Record<AdminResourceKey, readonly string[]>;

export type AdminRepositoryErrorCode =
  | "CONFLICT"
  | "NOT_FOUND"
  | "PROFILE_REQUIRED"
  | "READ_ONLY";

export class AdminRepositoryError extends Error {
  constructor(
    readonly code: AdminRepositoryErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "AdminRepositoryError";
  }
}

function isAdminFieldValue(value: unknown): value is AdminFieldValue {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    (Array.isArray(value) && value.every((entry) => typeof entry === "string"))
  );
}

function toAdminFieldValue(value: unknown): AdminFieldValue | undefined {
  if (isAdminFieldValue(value)) return value;

  if (
    typeof value === "object" &&
    value !== null &&
    "toString" in value &&
    typeof value.toString === "function"
  ) {
    return value.toString();
  }

  return undefined;
}

function valuesForResource(
  resource: AdminResourceKey,
  row: Record<string, unknown>,
) {
  const values: Record<string, AdminFieldValue> = {};
  for (const field of editableFields[resource]) {
    const value = toAdminFieldValue(row[field]);
    if (value !== undefined) values[field] = value;
  }
  return values;
}

function titleForResource(
  resource: AdminResourceKey,
  row: Record<string, unknown>,
) {
  switch (resource) {
    case "profile":
      return String(row.name ?? "Profile");
    case "social-links":
      return String(row.label ?? "Social link");
    case "research-interests":
    case "skill-categories":
    case "skills":
    case "certifications":
      return String(row.name ?? "Untitled");
    case "research-projects":
    case "projects":
    case "publications":
    case "articles":
    case "resumes":
      return String(row.title ?? "Untitled");
    case "experience":
      return String(row.role ?? "Experience");
    case "education":
      return String(row.degree ?? "Education");
    case "site-settings":
      return "Site setting";
  }
}

function subtitleForResource(
  resource: AdminResourceKey,
  row: Record<string, unknown>,
) {
  const value = (() => {
    switch (resource) {
      case "profile":
        return row.headline;
      case "social-links":
        return row.url;
      case "research-interests":
      case "skill-categories":
        return row.description;
      case "research-projects":
      case "projects":
        return row.summary;
      case "experience":
        return row.organization;
      case "education":
        return row.institution;
      case "publications":
        return row.venue;
      case "articles":
        return row.excerpt;
      case "resumes":
        return row.version;
      case "skills":
        return row.categoryName;
      case "certifications":
        return row.issuer;
      case "site-settings":
        return undefined;
    }
  })();

  return typeof value === "string" && value ? value : undefined;
}

function toRecordSummary(
  resource: AdminResourceKey,
  row: Record<string, unknown>,
): AdminRecordSummaryDto {
  const updatedAt = row.updatedAt;
  if (!(updatedAt instanceof Date)) {
    throw new Error("Admin record is missing its update timestamp.");
  }

  return {
    id: String(row.id),
    title: titleForResource(resource, row),
    ...(subtitleForResource(resource, row)
      ? { subtitle: subtitleForResource(resource, row) }
      : {}),
    ...(typeof row.status === "string" ? { status: row.status } : {}),
    ...(typeof row.featured === "boolean" ? { featured: row.featured } : {}),
    ...(typeof row.sortOrder === "number" ? { sortOrder: row.sortOrder } : {}),
    updatedAt: updatedAt.toISOString(),
  };
}

function toRecord(
  resource: AdminResourceKey,
  row: Record<string, unknown>,
): AdminRecordDto {
  return {
    ...toRecordSummary(resource, row),
    values: valuesForResource(resource, row),
  };
}

async function findPrimaryProfile() {
  return getDatabase().profile.findUnique({
    where: { key: PRIMARY_PROFILE_KEY },
    select: { id: true, status: true },
  });
}

async function requirePrimaryProfile() {
  const profile = await findPrimaryProfile();
  if (!profile) {
    throw new AdminRepositoryError(
      "PROFILE_REQUIRED",
      "Create the primary profile before managing related portfolio content.",
    );
  }
  return profile;
}

export async function loadAdminProfileExists(): Promise<boolean> {
  return Boolean(await findPrimaryProfile());
}

function publishedAtForCreate(status: string) {
  return resolvePublishedAt(
    status as "ARCHIVED" | AdminVisibilityStatus,
    null,
  );
}

function publishedAtForUpdate(status: string | undefined, current: Date | null) {
  if (status === undefined) return {};
  return {
    publishedAt: resolvePublishedAt(
      status as "ARCHIVED" | AdminVisibilityStatus,
      current,
    ),
  };
}

function notFound(resource: AdminResourceKey): never {
  throw new AdminRepositoryError(
    "NOT_FOUND",
    `The requested ${resource.replaceAll("-", " ")} record was not found.`,
  );
}

function statusBreakdown(rows: { status: string; _count: { _all: number } }[]) {
  const byStatus = new Map(rows.map((row) => [row.status, row._count._all]));
  return {
    count: rows.reduce((total, row) => total + row._count._all, 0),
    draftCount: byStatus.get(ContentStatus.DRAFT) ?? 0,
    publishedCount: byStatus.get(ContentStatus.PUBLISHED) ?? 0,
  };
}

type NestedParentRow = {
  id: string;
  title: string;
  status: string;
  source: string;
  publishedAt: Date | null;
  updatedAt: Date;
};

function toNestedParent(row: NestedParentRow): AdminNestedParentDto {
  return {
    id: row.id,
    title: row.title,
    status: row.status as AdminNestedParentDto["status"],
    source: row.source as AdminNestedParentDto["source"],
    publishedAt: row.publishedAt?.toISOString() ?? null,
    updatedAt: row.updatedAt.toISOString(),
  };
}

async function loadResearchNestedContent(
  parentId: string,
): Promise<AdminResearchNestedContentDto | null> {
  const profile = await requirePrimaryProfile();
  const row = await getDatabase().researchProject.findFirst({
    where: { id: parentId, profileId: profile.id },
    select: {
      id: true,
      title: true,
      status: true,
      source: true,
      publishedAt: true,
      updatedAt: true,
      questions: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          id: true,
          key: true,
          question: true,
          sortOrder: true,
        },
      },
    },
  });
  if (!row) return null;
  return {
    resource: "research-projects",
    parent: toNestedParent(row),
    questions: row.questions,
  };
}

async function loadProjectNestedContent(
  parentId: string,
): Promise<AdminProjectNestedContentDto | null> {
  const profile = await requirePrimaryProfile();
  const row = await getDatabase().project.findFirst({
    where: { id: parentId, profileId: profile.id },
    select: {
      id: true,
      title: true,
      status: true,
      source: true,
      publishedAt: true,
      updatedAt: true,
      contributions: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          id: true,
          key: true,
          label: true,
          description: true,
          sortOrder: true,
        },
      },
      features: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          id: true,
          key: true,
          name: true,
          description: true,
          sortOrder: true,
        },
      },
      technologies: {
        orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
        select: {
          id: true,
          slug: true,
          name: true,
          sortOrder: true,
        },
      },
    },
  });
  if (!row) return null;
  return {
    resource: "projects",
    parent: toNestedParent(row),
    contributions: row.contributions,
    features: row.features,
    technologies: row.technologies,
  };
}

export async function loadAdminNestedContent(
  resource: AdminNestedResourceKey,
  parentId: string,
): Promise<AdminNestedContentDto | null> {
  return resource === "research-projects"
    ? loadResearchNestedContent(parentId)
    : loadProjectNestedContent(parentId);
}

type ExistingNestedIdentity = {
  id: string;
};

type PlannedNestedEntry<Input, Existing> = {
  input: Input & { sortOrder: number };
  existing: Existing | null;
};

type NestedCollectionPlan<Input, Existing> = {
  entries: Array<PlannedNestedEntry<Input, Existing>>;
  removals: Existing[];
};

function planNestedCollection<
  Input extends { id?: string },
  Existing extends ExistingNestedIdentity,
>(
  existingRows: readonly Existing[],
  inputs: readonly Input[],
  inputNaturalKey: (input: Input) => string,
  existingNaturalKey: (row: Existing) => string,
): NestedCollectionPlan<Input, Existing> {
  const existingById = new Map(existingRows.map((row) => [row.id, row]));
  const existingByNaturalKey = new Map(
    existingRows.map((row) => [existingNaturalKey(row), row]),
  );
  const explicitlySubmittedIds = new Set(
    inputs.flatMap((input) => (input.id ? [input.id] : [])),
  );
  const claimedIds = new Set<string>();

  const entries = withCanonicalSortOrder(inputs).map((input) => {
    const existing = input.id
      ? existingById.get(input.id)
      : (() => {
          const naturalKeyMatch = existingByNaturalKey.get(
            inputNaturalKey(input),
          );
          return naturalKeyMatch &&
            explicitlySubmittedIds.has(naturalKeyMatch.id)
            ? undefined
            : naturalKeyMatch;
        })();

    if (input.id && !existing) {
      throw new AdminRepositoryError(
        "CONFLICT",
        "Nested content changed in another session. Refresh before saving.",
      );
    }
    if (existing && claimedIds.has(existing.id)) {
      throw new AdminRepositoryError(
        "CONFLICT",
        "A nested record cannot be submitted more than once.",
      );
    }
    if (existing) claimedIds.add(existing.id);

    return { input, existing: existing ?? null };
  });

  return {
    entries,
    removals: existingRows.filter((row) => !claimedIds.has(row.id)),
  };
}

function assertNestedParentState(
  revision: string,
  parent: { status: string; publishedAt: Date | null; updatedAt: Date },
) {
  if (!revisionMatches(revision, parent.updatedAt)) {
    throw new AdminRepositoryError(
      "CONFLICT",
      "This record changed in another session. Refresh before saving.",
    );
  }
  if (
    !isPublicationStateValid(
      parent.status as "ARCHIVED" | AdminVisibilityStatus,
      parent.publishedAt,
    )
  ) {
    throw new AdminRepositoryError(
      "CONFLICT",
      "Repair the parent record's visibility state before editing nested content.",
    );
  }
}

type NestedRemoval = { id: string; label: string };

function removalConfirmation(
  actualRemovals: readonly NestedRemoval[],
  suppliedRemovedIds: readonly string[],
  confirmed: true | undefined,
): AdminNestedRemovalConfirmation | null {
  const decision = decideAdminRemoval(
    actualRemovals.map(({ id }) => id),
    suppliedRemovedIds,
    confirmed === true,
  );
  if (decision === "mismatch") {
    throw new AdminRepositoryError(
      "CONFLICT",
      "The removal set is stale or incomplete. Refresh before saving.",
    );
  }
  if (decision === "proceed") return null;
  return {
    removedIds: actualRemovals.map(({ id }) => id),
    labels: actualRemovals.map(({ label }) => label),
  };
}

function claimedParentTimestamp(current: Date) {
  return new Date(Math.max(Date.now(), current.getTime() + 1));
}

const nestedTransactionOptions = {
  maxWait: 10_000,
  timeout: 60_000,
} as const;

export type AdminNestedSaveOutcome =
  | { kind: "saved"; content: AdminNestedContentDto }
  | {
      kind: "confirmation-required";
      confirmation: AdminNestedRemovalConfirmation;
    };

export async function loadAdminDashboardData(): Promise<AdminDashboardDto> {
  const database = getDatabase();
  const profile = await findPrimaryProfile();

  if (!profile) {
    return {
      profileExists: false,
      totalRecords: 0,
      resources: adminResourceKeys.map((resource) => ({
        resource,
        count: 0,
        draftCount: 0,
        publishedCount: 0,
      })),
    };
  }

  const [
    socialLinks,
    researchInterests,
    researchProjects,
    projects,
    experience,
    education,
    publications,
    articles,
    resumes,
    skillCategories,
    skills,
    certifications,
  ] = await Promise.all([
    database.socialLink.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.researchInterest.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.researchProject.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.project.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.experience.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.education.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.publication.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.article.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.resume.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.skillCategory.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
    database.skill.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { category: { profileId: profile.id } },
    }),
    database.certification.groupBy({
      by: ["status"],
      _count: { _all: true },
      where: { profileId: profile.id },
    }),
  ]);

  const counts = {
    profile: {
      count: 1,
      draftCount: profile.status === ContentStatus.DRAFT ? 1 : 0,
      publishedCount: profile.status === ContentStatus.PUBLISHED ? 1 : 0,
    },
    "social-links": statusBreakdown(socialLinks),
    "research-interests": statusBreakdown(researchInterests),
    "research-projects": statusBreakdown(researchProjects),
    projects: statusBreakdown(projects),
    experience: statusBreakdown(experience),
    education: statusBreakdown(education),
    publications: statusBreakdown(publications),
    articles: statusBreakdown(articles),
    resumes: statusBreakdown(resumes),
    "skill-categories": statusBreakdown(skillCategories),
    skills: statusBreakdown(skills),
    certifications: statusBreakdown(certifications),
    "site-settings": { count: 0, draftCount: 0, publishedCount: 0 },
  } satisfies Record<
    AdminResourceKey,
    { count: number; draftCount: number; publishedCount: number }
  >;

  const resources = adminResourceKeys.map((resource) => ({
    resource,
    ...counts[resource],
  }));

  return {
    profileExists: true,
    totalRecords: resources.reduce((total, entry) => total + entry.count, 0),
    resources,
  };
}

export async function loadAdminCollection(
  resource: AdminResourceKey,
): Promise<AdminRecordSummaryDto[]> {
  const database = getDatabase();
  const profile = await findPrimaryProfile();
  if (!profile || resource === "site-settings") return [];

  const orderBy = [{ sortOrder: "asc" as const }, { updatedAt: "desc" as const }];
  switch (resource) {
    case "profile": {
      const row = await database.profile.findUnique({
        where: { key: PRIMARY_PROFILE_KEY },
      });
      return row ? [toRecordSummary(resource, row)] : [];
    }
    case "social-links":
      return (
        await database.socialLink.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "research-interests":
      return (
        await database.researchInterest.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "research-projects":
      return (
        await database.researchProject.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "projects":
      return (
        await database.project.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "experience":
      return (
        await database.experience.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "education":
      return (
        await database.education.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "publications":
      return (
        await database.publication.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "articles":
      return (
        await database.article.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "resumes":
      return (
        await database.resume.findMany({
          where: { profileId: profile.id },
          orderBy: [{ isCurrent: "desc" }, { updatedAt: "desc" }],
        })
      ).map((row) => toRecordSummary(resource, row));
    case "skill-categories":
      return (
        await database.skillCategory.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
    case "skills":
      return (
        await database.skill.findMany({
          where: { category: { profileId: profile.id } },
          include: { category: { select: { name: true } } },
          orderBy,
        })
      ).map((row) =>
        toRecordSummary(resource, {
          ...row,
          categoryName: row.category.name,
        }),
      );
    case "certifications":
      return (
        await database.certification.findMany({
          where: { profileId: profile.id },
          orderBy,
        })
      ).map((row) => toRecordSummary(resource, row));
  }
}

export async function loadAdminRecord(
  resource: AdminResourceKey,
  id: string,
): Promise<AdminRecordDto | null> {
  const database = getDatabase();
  const profile = await findPrimaryProfile();
  if (!profile || resource === "site-settings") return null;

  switch (resource) {
    case "profile": {
      const row = await database.profile.findFirst({
        where: { id, key: PRIMARY_PROFILE_KEY },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "social-links": {
      const row = await database.socialLink.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "research-interests": {
      const row = await database.researchInterest.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "research-projects": {
      const row = await database.researchProject.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "projects": {
      const row = await database.project.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "experience": {
      const row = await database.experience.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "education": {
      const row = await database.education.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "publications": {
      const row = await database.publication.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "articles": {
      const row = await database.article.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "resumes": {
      const row = await database.resume.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "skill-categories": {
      const row = await database.skillCategory.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
    case "skills": {
      const row = await database.skill.findFirst({
        where: { id, category: { profileId: profile.id } },
        include: { category: { select: { name: true } } },
      });
      return row
        ? toRecord(resource, { ...row, categoryName: row.category.name })
        : null;
    }
    case "certifications": {
      const row = await database.certification.findFirst({
        where: { id, profileId: profile.id },
      });
      return row ? toRecord(resource, row) : null;
    }
  }
}

export async function createAdminRecord<Resource extends AdminResourceKey>(
  resource: Resource,
  input: AdminCreateInputMap[Resource],
): Promise<AdminRecordDto> {
  const database = getDatabase();

  switch (resource) {
    case "profile": {
      const data = input as AdminCreateInputMap["profile"];
      const existing = await findPrimaryProfile();
      if (existing) {
        throw new AdminRepositoryError(
          "CONFLICT",
          "The primary profile already exists. Edit it instead.",
        );
      }
      const row = await database.profile.create({
        data: {
          ...data,
          key: PRIMARY_PROFILE_KEY,
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "site-settings":
      throw new AdminRepositoryError(
        "READ_ONLY",
        "Site settings are read-only until a key is explicitly allowlisted.",
      );
    default:
      break;
  }

  const profile = await requirePrimaryProfile();
  switch (resource) {
    case "social-links": {
      const data = input as AdminCreateInputMap["social-links"];
      const row = await database.socialLink.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-interests": {
      const data = input as AdminCreateInputMap["research-interests"];
      const row = await database.researchInterest.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-projects": {
      const data = input as AdminCreateInputMap["research-projects"];
      const row = await database.researchProject.create({
        data: {
          ...data,
          profileId: profile.id,
          source: ContentSource.ADMIN,
          publishedAt: publishedAtForCreate(data.status),
        },
      });
      return toRecord(resource, row);
    }
    case "projects": {
      const data = input as AdminCreateInputMap["projects"];
      const row = await database.project.create({
        data: {
          ...data,
          profileId: profile.id,
          source: ContentSource.ADMIN,
          publishedAt: publishedAtForCreate(data.status),
        },
      });
      return toRecord(resource, row);
    }
    case "experience": {
      const data = input as AdminCreateInputMap["experience"];
      const row = await database.experience.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "education": {
      const data = input as AdminCreateInputMap["education"];
      const row = await database.education.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "publications": {
      const data = input as AdminCreateInputMap["publications"];
      const row = await database.publication.create({
        data: {
          ...data,
          profileId: profile.id,
          source: ContentSource.ADMIN,
          publishedAt: publishedAtForCreate(data.status),
        },
      });
      return toRecord(resource, row);
    }
    case "articles": {
      const data = input as AdminCreateInputMap["articles"];
      const row = await database.article.create({
        data: {
          ...data,
          profileId: profile.id,
          source: ContentSource.ADMIN,
          publishedAt: publishedAtForCreate(data.status),
        },
      });
      return toRecord(resource, row);
    }
    case "resumes": {
      const data = input as AdminCreateInputMap["resumes"];
      const row = await database.$transaction(async (transaction) => {
        if (data.isCurrent) {
          await transaction.resume.updateMany({
            where: { profileId: profile.id, isCurrent: true },
            data: { isCurrent: false },
          });
        }
        return transaction.resume.create({
          data: {
            ...data,
            profileId: profile.id,
            source: ContentSource.ADMIN,
            publishedAt: publishedAtForCreate(data.status),
          },
        });
      });
      return toRecord(resource, row);
    }
    case "skill-categories": {
      const data = input as AdminCreateInputMap["skill-categories"];
      const row = await database.skillCategory.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "skills": {
      const data = input as AdminCreateInputMap["skills"];
      const category = await database.skillCategory.findFirst({
        where: {
          id: data.categoryId,
          profileId: profile.id,
          status: { not: ContentStatus.ARCHIVED },
        },
        select: { id: true, name: true },
      });
      if (!category) notFound("skill-categories");
      const row = await database.skill.create({
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, { ...row, categoryName: category.name });
    }
    case "certifications": {
      const data = input as AdminCreateInputMap["certifications"];
      const row = await database.certification.create({
        data: { ...data, profileId: profile.id, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
  }

  throw new AdminRepositoryError("READ_ONLY", "This operation is unavailable.");
}

export async function updateAdminRecord<Resource extends AdminResourceKey>(
  resource: Resource,
  id: string,
  input: AdminUpdateInputMap[Resource],
): Promise<AdminRecordDto> {
  const database = getDatabase();

  if (resource === "site-settings") {
    throw new AdminRepositoryError(
      "READ_ONLY",
      "Site settings are read-only until a key is explicitly allowlisted.",
    );
  }

  const profile = await requirePrimaryProfile();
  switch (resource) {
    case "profile": {
      const data = input as AdminUpdateInputMap["profile"];
      const existing = await database.profile.findFirst({
        where: { id, key: PRIMARY_PROFILE_KEY },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.profile.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "social-links": {
      const data = input as AdminUpdateInputMap["social-links"];
      const existing = await database.socialLink.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.socialLink.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-interests": {
      const data = input as AdminUpdateInputMap["research-interests"];
      const existing = await database.researchInterest.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.researchInterest.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-projects": {
      const data = input as AdminUpdateInputMap["research-projects"];
      const existing = await database.researchProject.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.researchProject.update({
        where: { id: existing.id },
        data: {
          ...data,
          ...publishedAtForUpdate(data.status, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "projects": {
      const data = input as AdminUpdateInputMap["projects"];
      const existing = await database.project.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.project.update({
        where: { id: existing.id },
        data: {
          ...data,
          ...publishedAtForUpdate(data.status, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "experience": {
      const data = input as AdminUpdateInputMap["experience"];
      const existing = await database.experience.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.experience.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "education": {
      const data = input as AdminUpdateInputMap["education"];
      const existing = await database.education.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.education.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "publications": {
      const data = input as AdminUpdateInputMap["publications"];
      const existing = await database.publication.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.publication.update({
        where: { id: existing.id },
        data: {
          ...data,
          ...publishedAtForUpdate(data.status, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "articles": {
      const data = input as AdminUpdateInputMap["articles"];
      const existing = await database.article.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.article.update({
        where: { id: existing.id },
        data: {
          ...data,
          ...publishedAtForUpdate(data.status, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "resumes": {
      const data = input as AdminUpdateInputMap["resumes"];
      const row = await database.$transaction(async (transaction) => {
        const existing = await transaction.resume.findFirst({
          where: { id, profileId: profile.id },
          select: { id: true, publishedAt: true },
        });
        if (!existing) notFound(resource);
        if (data.isCurrent) {
          await transaction.resume.updateMany({
            where: { profileId: profile.id, isCurrent: true, id: { not: id } },
            data: { isCurrent: false },
          });
        }
        return transaction.resume.update({
          where: { id: existing.id },
          data: {
            ...data,
            ...publishedAtForUpdate(data.status, existing.publishedAt),
            source: ContentSource.ADMIN,
          },
        });
      });
      return toRecord(resource, row);
    }
    case "skill-categories": {
      const data = input as AdminUpdateInputMap["skill-categories"];
      const existing = await database.skillCategory.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.skillCategory.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "skills": {
      const data = input as AdminUpdateInputMap["skills"];
      const existing = await database.skill.findFirst({
        where: { id, category: { profileId: profile.id } },
        select: { id: true, categoryId: true },
      });
      if (!existing) notFound(resource);
      const categoryId = data.categoryId ?? existing.categoryId;
      const category = await database.skillCategory.findFirst({
        where: {
          id: categoryId,
          profileId: profile.id,
          status: { not: ContentStatus.ARCHIVED },
        },
        select: { id: true, name: true },
      });
      if (!category) notFound("skill-categories");
      const row = await database.skill.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, { ...row, categoryName: category.name });
    }
    case "certifications": {
      const data = input as AdminUpdateInputMap["certifications"];
      const existing = await database.certification.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.certification.update({
        where: { id: existing.id },
        data: { ...data, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
  }

  throw new AdminRepositoryError("READ_ONLY", "This operation is unavailable.");
}

async function saveResearchNestedContent(
  parentId: string,
  input: AdminNestedContentInputMap["research-projects"],
): Promise<AdminNestedSaveOutcome> {
  return getDatabase().$transaction(async (transaction) => {
    const profile = await transaction.profile.findUnique({
      where: { key: PRIMARY_PROFILE_KEY },
      select: { id: true },
    });
    if (!profile) {
      throw new AdminRepositoryError(
        "PROFILE_REQUIRED",
        "Create the primary profile before managing related portfolio content.",
      );
    }

    const parent = await transaction.researchProject.findFirst({
      where: { id: parentId, profileId: profile.id },
      select: {
        id: true,
        status: true,
        publishedAt: true,
        updatedAt: true,
        questions: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            question: true,
            sortOrder: true,
          },
        },
      },
    });
    if (!parent) notFound("research-projects");
    assertNestedParentState(input.revision, parent);

    const plan = planNestedCollection(
      parent.questions,
      input.questions,
      (question) => question.key,
      (question) => question.key,
    );
    const confirmation = removalConfirmation(
      plan.removals.map((question) => ({
        id: question.id,
        label: `Research question: ${question.question}`,
      })),
      input.removedIds,
      input.confirmRemoval,
    );
    if (confirmation) {
      return { kind: "confirmation-required", confirmation };
    }

    const claimedAt = claimedParentTimestamp(parent.updatedAt);
    const claimed = await transaction.researchProject.updateMany({
      where: {
        id: parent.id,
        profileId: profile.id,
        updatedAt: parent.updatedAt,
      },
      data: { source: ContentSource.ADMIN, updatedAt: claimedAt },
    });
    if (claimed.count !== 1) {
      throw new AdminRepositoryError(
        "CONFLICT",
        "This record changed in another session. Refresh before saving.",
      );
    }

    if (plan.removals.length > 0) {
      await transaction.researchQuestion.deleteMany({
        where: {
          researchProjectId: parent.id,
          id: { in: plan.removals.map(({ id }) => id) },
        },
      });
    }
    for (const entry of plan.entries) {
      if (entry.existing && entry.existing.key !== entry.input.key) {
        await transaction.researchQuestion.update({
          where: { id: entry.existing.id },
          data: { key: `__admin_sync__${entry.existing.id}` },
        });
      }
    }
    for (const entry of plan.entries) {
      const data = {
        key: entry.input.key,
        question: entry.input.question,
        sortOrder: entry.input.sortOrder,
      };
      if (entry.existing) {
        await transaction.researchQuestion.update({
          where: { id: entry.existing.id },
          data,
        });
      } else {
        await transaction.researchQuestion.create({
          data: { ...data, researchProjectId: parent.id },
        });
      }
    }

    const saved = await transaction.researchProject.findUnique({
      where: { id: parent.id },
      select: {
        id: true,
        title: true,
        status: true,
        source: true,
        publishedAt: true,
        updatedAt: true,
        questions: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            question: true,
            sortOrder: true,
          },
        },
      },
    });
    if (!saved) notFound("research-projects");
    return {
      kind: "saved",
      content: {
        resource: "research-projects",
        parent: toNestedParent(saved),
        questions: saved.questions,
      },
    };
  }, nestedTransactionOptions);
}

async function saveProjectNestedContent(
  parentId: string,
  input: AdminNestedContentInputMap["projects"],
): Promise<AdminNestedSaveOutcome> {
  return getDatabase().$transaction(async (transaction) => {
    const profile = await transaction.profile.findUnique({
      where: { key: PRIMARY_PROFILE_KEY },
      select: { id: true },
    });
    if (!profile) {
      throw new AdminRepositoryError(
        "PROFILE_REQUIRED",
        "Create the primary profile before managing related portfolio content.",
      );
    }

    const parent = await transaction.project.findFirst({
      where: { id: parentId, profileId: profile.id },
      select: {
        id: true,
        status: true,
        publishedAt: true,
        updatedAt: true,
        contributions: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            label: true,
            description: true,
            sortOrder: true,
          },
        },
        features: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            name: true,
            description: true,
            sortOrder: true,
          },
        },
        technologies: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            slug: true,
            name: true,
            sortOrder: true,
          },
        },
      },
    });
    if (!parent) notFound("projects");
    assertNestedParentState(input.revision, parent);

    const contributionPlan = planNestedCollection(
      parent.contributions,
      input.contributions,
      (contribution) => contribution.key,
      (contribution) => contribution.key,
    );
    const featurePlan = planNestedCollection(
      parent.features,
      input.features,
      (feature) => feature.key,
      (feature) => feature.key,
    );
    const technologyPlan = planNestedCollection(
      parent.technologies,
      input.technologies,
      (technology) => technology.slug,
      (technology) => technology.slug,
    );
    const confirmation = removalConfirmation(
      [
        ...contributionPlan.removals.map((contribution) => ({
          id: contribution.id,
          label: `Contribution: ${contribution.label}`,
        })),
        ...featurePlan.removals.map((feature) => ({
          id: feature.id,
          label: `Feature: ${feature.name}`,
        })),
        ...technologyPlan.removals.map((technology) => ({
          id: technology.id,
          label: `Technology: ${technology.name}`,
        })),
      ],
      input.removedIds,
      input.confirmRemoval,
    );
    if (confirmation) {
      return { kind: "confirmation-required", confirmation };
    }

    const claimedAt = claimedParentTimestamp(parent.updatedAt);
    const claimed = await transaction.project.updateMany({
      where: {
        id: parent.id,
        profileId: profile.id,
        updatedAt: parent.updatedAt,
      },
      data: { source: ContentSource.ADMIN, updatedAt: claimedAt },
    });
    if (claimed.count !== 1) {
      throw new AdminRepositoryError(
        "CONFLICT",
        "This record changed in another session. Refresh before saving.",
      );
    }

    if (contributionPlan.removals.length > 0) {
      await transaction.projectContribution.deleteMany({
        where: {
          projectId: parent.id,
          id: { in: contributionPlan.removals.map(({ id }) => id) },
        },
      });
    }
    if (featurePlan.removals.length > 0) {
      await transaction.projectFeature.deleteMany({
        where: {
          projectId: parent.id,
          id: { in: featurePlan.removals.map(({ id }) => id) },
        },
      });
    }
    if (technologyPlan.removals.length > 0) {
      await transaction.projectTechnology.deleteMany({
        where: {
          projectId: parent.id,
          id: { in: technologyPlan.removals.map(({ id }) => id) },
        },
      });
    }

    for (const entry of contributionPlan.entries) {
      if (entry.existing && entry.existing.key !== entry.input.key) {
        await transaction.projectContribution.update({
          where: { id: entry.existing.id },
          data: { key: `__admin_sync__${entry.existing.id}` },
        });
      }
    }
    for (const entry of featurePlan.entries) {
      if (entry.existing && entry.existing.key !== entry.input.key) {
        await transaction.projectFeature.update({
          where: { id: entry.existing.id },
          data: { key: `__admin_sync__${entry.existing.id}` },
        });
      }
    }
    for (const entry of technologyPlan.entries) {
      if (entry.existing && entry.existing.slug !== entry.input.slug) {
        await transaction.projectTechnology.update({
          where: { id: entry.existing.id },
          data: { slug: `__admin_sync__${entry.existing.id}` },
        });
      }
    }

    for (const entry of contributionPlan.entries) {
      const data = {
        key: entry.input.key,
        label: entry.input.label,
        description: entry.input.description,
        sortOrder: entry.input.sortOrder,
      };
      if (entry.existing) {
        await transaction.projectContribution.update({
          where: { id: entry.existing.id },
          data,
        });
      } else {
        await transaction.projectContribution.create({
          data: { ...data, projectId: parent.id },
        });
      }
    }
    for (const entry of featurePlan.entries) {
      const data = {
        key: entry.input.key,
        name: entry.input.name,
        description: entry.input.description ?? null,
        sortOrder: entry.input.sortOrder,
      };
      if (entry.existing) {
        await transaction.projectFeature.update({
          where: { id: entry.existing.id },
          data,
        });
      } else {
        await transaction.projectFeature.create({
          data: { ...data, projectId: parent.id },
        });
      }
    }
    for (const entry of technologyPlan.entries) {
      const data = {
        slug: entry.input.slug,
        name: entry.input.name,
        sortOrder: entry.input.sortOrder,
      };
      if (entry.existing) {
        await transaction.projectTechnology.update({
          where: { id: entry.existing.id },
          data,
        });
      } else {
        await transaction.projectTechnology.create({
          data: { ...data, projectId: parent.id },
        });
      }
    }

    const saved = await transaction.project.findUnique({
      where: { id: parent.id },
      select: {
        id: true,
        title: true,
        status: true,
        source: true,
        publishedAt: true,
        updatedAt: true,
        contributions: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            label: true,
            description: true,
            sortOrder: true,
          },
        },
        features: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            key: true,
            name: true,
            description: true,
            sortOrder: true,
          },
        },
        technologies: {
          orderBy: [{ sortOrder: "asc" }, { id: "asc" }],
          select: {
            id: true,
            slug: true,
            name: true,
            sortOrder: true,
          },
        },
      },
    });
    if (!saved) notFound("projects");
    return {
      kind: "saved",
      content: {
        resource: "projects",
        parent: toNestedParent(saved),
        contributions: saved.contributions,
        features: saved.features,
        technologies: saved.technologies,
      },
    };
  }, nestedTransactionOptions);
}

export async function saveAdminNestedContent<
  Resource extends AdminNestedResourceKey,
>(
  resource: Resource,
  parentId: string,
  input: AdminNestedContentInputMap[Resource],
): Promise<AdminNestedSaveOutcome> {
  if (resource === "research-projects") {
    return saveResearchNestedContent(
      parentId,
      input as AdminNestedContentInputMap["research-projects"],
    );
  }
  return saveProjectNestedContent(
    parentId,
    input as AdminNestedContentInputMap["projects"],
  );
}

export async function setAdminRecordVisibility(
  resource: AdminResourceKey,
  id: string,
  visibility: AdminVisibilityStatus,
): Promise<AdminRecordDto> {
  if (resource === "site-settings") {
    throw new AdminRepositoryError(
      "READ_ONLY",
      "Site settings do not support visibility changes.",
    );
  }
  const database = getDatabase();
  const profile = await requirePrimaryProfile();

  switch (resource) {
    case "profile": {
      if (id !== profile.id) notFound(resource);
      const row = await database.profile.update({
        where: { id: profile.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "social-links": {
      const existing = await database.socialLink.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.socialLink.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-interests": {
      const existing = await database.researchInterest.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.researchInterest.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "research-projects": {
      const existing = await database.researchProject.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.researchProject.update({
        where: { id: existing.id },
        data: {
          status: visibility,
          publishedAt: resolvePublishedAt(visibility, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "projects": {
      const existing = await database.project.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.project.update({
        where: { id: existing.id },
        data: {
          status: visibility,
          publishedAt: resolvePublishedAt(visibility, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "experience": {
      const existing = await database.experience.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.experience.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "education": {
      const existing = await database.education.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.education.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "publications": {
      const existing = await database.publication.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.publication.update({
        where: { id: existing.id },
        data: {
          status: visibility,
          publishedAt: resolvePublishedAt(visibility, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "articles": {
      const existing = await database.article.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.article.update({
        where: { id: existing.id },
        data: {
          status: visibility,
          publishedAt: resolvePublishedAt(visibility, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "resumes": {
      const existing = await database.resume.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, publishedAt: true },
      });
      if (!existing) notFound(resource);
      const row = await database.resume.update({
        where: { id: existing.id },
        data: {
          status: visibility,
          publishedAt: resolvePublishedAt(visibility, existing.publishedAt),
          source: ContentSource.ADMIN,
        },
      });
      return toRecord(resource, row);
    }
    case "skill-categories": {
      const existing = await database.skillCategory.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.skillCategory.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
    case "skills": {
      const existing = await database.skill.findFirst({
        where: { id, category: { profileId: profile.id } },
        select: { id: true, category: { select: { name: true } } },
      });
      if (!existing) notFound(resource);
      const row = await database.skill.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, {
        ...row,
        categoryName: existing.category.name,
      });
    }
    case "certifications": {
      const existing = await database.certification.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!existing) notFound(resource);
      const row = await database.certification.update({
        where: { id: existing.id },
        data: { status: visibility, source: ContentSource.ADMIN },
      });
      return toRecord(resource, row);
    }
  }

  throw new AdminRepositoryError("READ_ONLY", "This operation is unavailable.");
}

type DeleteOutcome = "archived";

export async function deleteAdminRecord(
  resource: AdminResourceKey,
  id: string,
): Promise<DeleteOutcome> {
  const database = getDatabase();
  // Phase 6 uses reversible archival for every status-bearing resource. This
  // protects verified content, avoids ambiguous cascade deletes, and promotes
  // archived seed records to ADMIN so a later explicit seed cannot restore them.
  if (resource === "profile") {
    throw new AdminRepositoryError(
      "READ_ONLY",
      "The primary profile cannot be deleted. Archive it instead.",
    );
  }
  if (resource === "site-settings") {
    throw new AdminRepositoryError(
      "READ_ONLY",
      "Site settings are read-only until a key is explicitly allowlisted.",
    );
  }

  const profile = await requirePrimaryProfile();
  switch (resource) {
    case "social-links": {
      const row = await database.socialLink.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.socialLink.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
    case "research-interests": {
      const row = await database.researchInterest.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.researchInterest.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
    case "research-projects": {
      const row = await database.researchProject.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.researchProject.update({
        where: { id: row.id },
        data: {
          status: ContentStatus.ARCHIVED,
          source: ContentSource.ADMIN,
          publishedAt: null,
        },
      });
      return "archived";
    }
    case "projects": {
      const row = await database.project.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.project.update({
        where: { id: row.id },
        data: {
          status: ContentStatus.ARCHIVED,
          source: ContentSource.ADMIN,
          publishedAt: null,
        },
      });
      return "archived";
    }
    case "experience": {
      const row = await database.experience.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.experience.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
    case "education": {
      const row = await database.education.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.education.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
    case "publications": {
      const row = await database.publication.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.publication.update({
        where: { id: row.id },
        data: {
          status: ContentStatus.ARCHIVED,
          source: ContentSource.ADMIN,
          publishedAt: null,
        },
      });
      return "archived";
    }
    case "articles": {
      const row = await database.article.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.article.update({
        where: { id: row.id },
        data: {
          status: ContentStatus.ARCHIVED,
          source: ContentSource.ADMIN,
          publishedAt: null,
        },
      });
      return "archived";
    }
    case "resumes": {
      const row = await database.resume.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.resume.update({
        where: { id: row.id },
        data: {
          status: ContentStatus.ARCHIVED,
          source: ContentSource.ADMIN,
          isCurrent: false,
          publishedAt: null,
        },
      });
      return "archived";
    }
    case "skill-categories": {
      const row = await database.skillCategory.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true },
      });
      if (!row) notFound(resource);
      await database.$transaction(async (transaction) => {
        const activeSkills = await transaction.skill.count({
          where: {
            categoryId: row.id,
            status: { not: ContentStatus.ARCHIVED },
          },
        });

        if (activeSkills > 0) {
          throw new AdminRepositoryError(
            "CONFLICT",
            "Archive or move the active skills in this category first.",
          );
        }

        await transaction.skillCategory.update({
          where: { id: row.id },
          data: {
            status: ContentStatus.ARCHIVED,
            source: ContentSource.ADMIN,
          },
        });
      });
      return "archived";
    }
    case "skills": {
      const row = await database.skill.findFirst({
        where: { id, category: { profileId: profile.id } },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.skill.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
    case "certifications": {
      const row = await database.certification.findFirst({
        where: { id, profileId: profile.id },
        select: { id: true, source: true },
      });
      if (!row) notFound(resource);
      await database.certification.update({
        where: { id: row.id },
        data: { status: ContentStatus.ARCHIVED, source: ContentSource.ADMIN },
      });
      return "archived";
    }
  }
}
