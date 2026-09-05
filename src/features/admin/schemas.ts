import { z } from "zod";

import {
  adminNestedResourceKeys,
  adminResourceKeys,
  type AdminNestedContentInputMap,
  type AdminNestedResourceKey,
  type AdminResourceKey,
} from "@/features/admin/types";

const EDITABLE_CONTENT_STATUSES = ["DRAFT", "PUBLISHED"] as const;
const SOCIAL_LINK_KINDS = [
  "EMAIL",
  "LINKEDIN",
  "GITHUB",
  "WEBSITE",
  "OTHER",
] as const;
const RESEARCH_STAGES = [
  "RESEARCH_INTEREST",
  "EXPLORING",
  "EARLY_STAGE",
  "IN_PROGRESS",
  "WORKING_PAPER",
  "SUBMITTED",
  "ACCEPTED",
  "PUBLISHED",
  "ARCHIVED",
] as const;
const PUBLICATION_STAGES = [
  "WORKING_PAPER",
  "SUBMITTED",
  "ACCEPTED",
  "PUBLISHED",
  "ARCHIVED",
] as const;
const PUBLICATION_TYPES = [
  "JOURNAL_ARTICLE",
  "CONFERENCE_PAPER",
  "WORKSHOP_PAPER",
  "PREPRINT",
  "BOOK_CHAPTER",
  "THESIS",
  "OTHER",
] as const;
const PROJECT_STATUSES = [
  "PLANNED",
  "IN_PROGRESS",
  "ON_HOLD",
  "COMPLETED",
  "ARCHIVED",
] as const;
const EXPERIENCE_TYPES = [
  "CLIENT_ENGAGEMENT",
  "EMPLOYMENT",
  "INTERNSHIP",
  "CONTRACT",
  "OTHER",
] as const;

function emptyToNull(value: unknown) {
  return typeof value === "string" && value.trim() === "" ? null : value;
}

function numberFromForm(value: unknown) {
  if (value === "" || value === null || value === undefined) return value;
  if (typeof value === "string" && /^-?\d+(?:\.\d+)?$/.test(value.trim())) {
    return Number(value);
  }
  return value;
}

function booleanFromForm(value: unknown) {
  if (value === "true" || value === "on" || value === "1") return true;
  if (value === "false" || value === "off" || value === "0") return false;
  return value;
}

function stringListFromForm(value: unknown) {
  if (value === null || value === "") return [];
  if (typeof value !== "string") return value;

  const trimmed = value.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) {
    try {
      return JSON.parse(trimmed) as unknown;
    } catch {
      return value;
    }
  }

  return trimmed.split(/\r?\n/).map((entry) => entry.trim());
}

function requiredText(label: string, maximum = 240) {
  return z
    .string({ error: `${label} is required.` })
    .trim()
    .min(1, `${label} is required.`)
    .max(maximum, `${label} must be ${maximum} characters or fewer.`);
}

function optionalText(maximum = 240) {
  return z.preprocess(
    emptyToNull,
    z.string().trim().max(maximum).nullable().optional(),
  );
}

const slugSchema = z
  .string({ error: "Slug is required." })
  .trim()
  .min(1, "Slug is required.")
  .max(128)
  .regex(
    /^[a-z0-9]+(?:-[a-z0-9]+)*$/,
    "Use lowercase letters, numbers, and single hyphens only.",
  );

const keySchema = z
  .string({ error: "Key is required." })
  .trim()
  .min(1, "Key is required.")
  .max(128)
  .regex(
    /^[a-z0-9]+(?:-[a-z0-9]+)*$/,
    "Use lowercase letters, numbers, and single hyphens only.",
  );

const idSchema = z.uuid("A valid record identifier is required.");

// Archival is intentionally excluded. Destructive state changes must flow
// through the separately confirmed archive action.
const statusSchema = z.enum(EDITABLE_CONTENT_STATUSES).default("DRAFT");
const sortOrderSchema = z.preprocess(
  numberFromForm,
  z.number().int().min(0).max(1_000_000).default(0),
);
const featuredSchema = z.preprocess(booleanFromForm, z.boolean().default(false));
const currentSchema = z.preprocess(booleanFromForm, z.boolean().default(false));
const monthSchema = z.preprocess(
  numberFromForm,
  z.number().int().min(1).max(12).nullable().optional(),
);
const yearSchema = z.preprocess(
  numberFromForm,
  z.number().int().min(1900).max(2200).nullable().optional(),
);
const stringListSchema = z.preprocess(
  stringListFromForm,
  z
    .array(z.string().trim().min(1).max(500))
    .max(100)
    .default([])
    .transform((entries) => [...new Set(entries)]),
);

type PeriodFields = {
  startYear?: number | null;
  startMonth?: number | null;
  endYear?: number | null;
  endMonth?: number | null;
  isCurrent?: boolean;
};

function validatePeriod(
  data: PeriodFields,
  context: z.RefinementCtx,
  partial: boolean,
) {
  if (
    data.startMonth != null &&
    (data.startYear === null || (!partial && data.startYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["startYear"],
      message: "Start year is required when a start month is provided.",
    });
  }
  if (
    data.endMonth != null &&
    (data.endYear === null || (!partial && data.endYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["endYear"],
      message: "End year is required when an end month is provided.",
    });
  }
  if (
    data.endYear != null &&
    (data.startYear === null || (!partial && data.startYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["startYear"],
      message: "Start year is required when an end year is provided.",
    });
  }
  if (data.startYear != null && data.endYear != null) {
    const endPrecedesStart =
      data.endYear < data.startYear ||
      (data.endYear === data.startYear &&
        data.startMonth != null &&
        data.endMonth != null &&
        data.endMonth < data.startMonth);
    if (endPrecedesStart) {
      context.addIssue({
        code: "custom",
        path: ["endYear"],
        message: "End date cannot be earlier than the start date.",
      });
    }
  }
  if (
    data.isCurrent === true &&
    (data.endYear != null || data.endMonth != null)
  ) {
    context.addIssue({
      code: "custom",
      path: ["isCurrent"],
      message: "Current records cannot also have an end date.",
    });
  }
}

type EducationFields = PeriodFields & {
  gpa?: number | null;
  gpaScale?: number | null;
};

function validateEducation(
  data: EducationFields,
  context: z.RefinementCtx,
  partial: boolean,
) {
  validatePeriod(data, context, partial);
  if (data.gpa != null && data.gpaScale != null && data.gpa > data.gpaScale) {
    context.addIssue({
      code: "custom",
      path: ["gpa"],
      message: "GPA cannot be greater than the GPA scale.",
    });
  }
}

type CertificationFields = {
  issueYear?: number | null;
  issueMonth?: number | null;
  expiryYear?: number | null;
  expiryMonth?: number | null;
  doesNotExpire?: boolean;
};

function validateCertification(
  data: CertificationFields,
  context: z.RefinementCtx,
  partial: boolean,
) {
  if (
    data.issueMonth != null &&
    (data.issueYear === null || (!partial && data.issueYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["issueYear"],
      message: "Issue year is required when an issue month is provided.",
    });
  }
  if (
    data.expiryMonth != null &&
    (data.expiryYear === null || (!partial && data.expiryYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["expiryYear"],
      message: "Expiry year is required when an expiry month is provided.",
    });
  }
  if (
    data.expiryYear != null &&
    (data.issueYear === null || (!partial && data.issueYear === undefined))
  ) {
    context.addIssue({
      code: "custom",
      path: ["issueYear"],
      message: "Issue year is required when an expiry year is provided.",
    });
  }
  if (data.issueYear != null && data.expiryYear != null) {
    const expiryPrecedesIssue =
      data.expiryYear < data.issueYear ||
      (data.expiryYear === data.issueYear &&
        data.issueMonth != null &&
        data.expiryMonth != null &&
        data.expiryMonth < data.issueMonth);
    if (expiryPrecedesIssue) {
      context.addIssue({
        code: "custom",
        path: ["expiryYear"],
        message: "Expiry date cannot be earlier than the issue date.",
      });
    }
  }
  if (
    data.doesNotExpire === true &&
    (data.expiryYear != null || data.expiryMonth != null)
  ) {
    context.addIssue({
      code: "custom",
      path: ["doesNotExpire"],
      message: "A non-expiring certification cannot have an expiry date.",
    });
  }
}

function optionalHttpUrl(label: string) {
  return z.preprocess(
    emptyToNull,
    z
      .string()
      .trim()
      .max(2_048)
      .url(`${label} must be a valid URL.`)
      .refine((value) => {
        const parsed = new URL(value);
        return (
          (parsed.protocol === "https:" || parsed.protocol === "http:") &&
          !parsed.username &&
          !parsed.password
        );
      }, `${label} must use HTTP or HTTPS and cannot contain credentials.`)
      .nullable()
      .optional(),
  );
}

const socialUrlSchema = z
  .string({ error: "URL is required." })
  .trim()
  .min(1, "URL is required.")
  .max(2_048)
  .url("Enter a valid URL.")
  .refine((value) => {
    const parsed = new URL(value);
    return (
      ["https:", "http:", "mailto:"].includes(parsed.protocol) &&
      !parsed.username &&
      !parsed.password
    );
  }, "Use HTTP, HTTPS, or mailto without embedded credentials.");

const profileShape = {
  slug: slugSchema,
  name: requiredText("Name"),
  positioning: stringListSchema,
  headline: requiredText("Headline", 1_000),
  introduction: requiredText("Introduction", 5_000),
  about: stringListSchema,
  location: optionalText(),
  status: statusSchema,
};

const socialLinkShape = {
  key: keySchema,
  kind: z.enum(SOCIAL_LINK_KINDS),
  label: requiredText("Label"),
  url: socialUrlSchema,
  handle: optionalText(),
  sortOrder: sortOrderSchema,
  status: statusSchema,
};

function validateSocialLinkProtocol(
  data: { kind: (typeof SOCIAL_LINK_KINDS)[number]; url: string },
  context: z.RefinementCtx,
) {
  const protocol = new URL(data.url).protocol;
  if (data.kind === "EMAIL" && protocol !== "mailto:") {
    context.addIssue({
      code: "custom",
      path: ["url"],
      message: "Email links must use a mailto: URL.",
    });
  }
  if (
    data.kind !== "EMAIL" &&
    protocol !== "https:" &&
    protocol !== "http:"
  ) {
    context.addIssue({
      code: "custom",
      path: ["url"],
      message: "Non-email links must use HTTP or HTTPS.",
    });
  }
}

const researchInterestShape = {
  slug: slugSchema,
  name: requiredText("Name"),
  description: optionalText(5_000),
  sortOrder: sortOrderSchema,
  status: statusSchema,
};

const researchProjectShape = {
  slug: slugSchema,
  title: requiredText("Title", 500),
  summary: requiredText("Summary", 5_000),
  abstract: optionalText(20_000),
  motivation: optionalText(10_000),
  methodology: optionalText(20_000),
  methodologySummary: optionalText(5_000),
  domain: optionalText(),
  researchArea: optionalText(),
  format: optionalText(),
  scopeBoundary: optionalText(10_000),
  evidenceStatus: optionalText(5_000),
  advisor: optionalText(),
  collaborators: stringListSchema,
  technologies: stringListSchema,
  datasets: stringListSchema,
  githubUrl: optionalHttpUrl("Repository URL"),
  paperUrl: optionalHttpUrl("Paper URL"),
  progressPercent: z.preprocess(
    numberFromForm,
    z.number().int().min(0).max(100).nullable().optional(),
  ),
  stage: z.enum(RESEARCH_STAGES),
  startYear: yearSchema,
  startMonth: monthSchema,
  endYear: yearSchema,
  endMonth: monthSchema,
  isCurrent: currentSchema,
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const projectShape = {
  slug: slugSchema,
  title: requiredText("Title", 500),
  shortTitle: optionalText(240),
  type: optionalText(240),
  category: optionalText(240),
  summary: requiredText("Summary", 5_000),
  description: optionalText(20_000),
  problemStatement: optionalText(10_000),
  motivation: optionalText(10_000),
  approach: optionalText(20_000),
  architecture: optionalText(20_000),
  aiModels: stringListSchema,
  datasets: stringListSchema,
  challenges: stringListSchema,
  results: stringListSchema,
  lessons: stringListSchema,
  institution: optionalText(500),
  advisor: optionalText(),
  role: optionalText(500),
  implementation: optionalText(2_000),
  repositoryUrl: optionalHttpUrl("Repository URL"),
  demoUrl: optionalHttpUrl("Demo URL"),
  documentationUrl: optionalHttpUrl("Documentation URL"),
  seoTitle: optionalText(240),
  seoDescription: optionalText(1_000),
  projectStatus: z.preprocess(
    emptyToNull,
    z.enum(PROJECT_STATUSES).nullable().optional(),
  ),
  startYear: yearSchema,
  startMonth: monthSchema,
  endYear: yearSchema,
  endMonth: monthSchema,
  isCurrent: currentSchema,
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const experienceShape = {
  slug: slugSchema,
  type: z.enum(EXPERIENCE_TYPES),
  organization: requiredText("Organization", 500),
  role: requiredText("Role", 500),
  location: optionalText(500),
  summary: requiredText("Summary", 10_000),
  domain: optionalText(500),
  practiceAreas: stringListSchema,
  highlights: stringListSchema,
  startYear: yearSchema,
  startMonth: monthSchema,
  endYear: yearSchema,
  endMonth: monthSchema,
  isCurrent: currentSchema,
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const educationShape = {
  slug: slugSchema,
  institution: requiredText("Institution", 500),
  degree: requiredText("Degree", 500),
  field: optionalText(500),
  location: optionalText(500),
  startYear: yearSchema,
  startMonth: monthSchema,
  endYear: yearSchema,
  endMonth: monthSchema,
  isCurrent: currentSchema,
  gpa: z.preprocess(
    numberFromForm,
    z.number().min(0).max(99.99).nullable().optional(),
  ),
  gpaScale: z.preprocess(
    numberFromForm,
    z.number().min(0.01).max(99.99).nullable().optional(),
  ),
  coursework: stringListSchema,
  thesis: optionalText(10_000),
  activities: stringListSchema,
  url: optionalHttpUrl("Education URL"),
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const publicationShape = {
  slug: slugSchema,
  title: requiredText("Title", 500),
  authors: stringListSchema.refine((authors) => authors.length > 0, {
    message: "Add at least one verified author.",
  }),
  venue: optionalText(500),
  type: z.enum(PUBLICATION_TYPES),
  year: yearSchema,
  abstract: optionalText(20_000),
  doi: optionalText(500),
  arxivUrl: optionalHttpUrl("arXiv URL"),
  paperUrl: optionalHttpUrl("Paper URL"),
  citation: optionalText(10_000),
  bibtex: optionalText(20_000),
  stage: z.enum(PUBLICATION_STAGES),
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const articleShape = {
  slug: slugSchema,
  title: requiredText("Title", 500),
  excerpt: optionalText(5_000),
  content: requiredText("Content", 200_000),
  contentFormat: z.literal("markdown").default("markdown"),
  category: optionalText(240),
  estimatedReadingMinutes: z.preprocess(
    numberFromForm,
    z.number().int().min(1).max(10_000).nullable().optional(),
  ),
  seoTitle: optionalText(240),
  seoDescription: optionalText(1_000),
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const resumeShape = {
  slug: slugSchema,
  title: requiredText("Title", 500),
  version: optionalText(120),
  isCurrent: currentSchema,
  status: statusSchema,
};

const skillCategoryShape = {
  slug: slugSchema,
  name: requiredText("Name", 500),
  description: optionalText(5_000),
  sortOrder: sortOrderSchema,
  status: statusSchema,
};

const skillShape = {
  categoryId: idSchema,
  slug: slugSchema,
  name: requiredText("Name", 500),
  description: optionalText(5_000),
  sortOrder: sortOrderSchema,
  status: statusSchema,
};

const certificationShape = {
  slug: slugSchema,
  name: requiredText("Name", 500),
  issuer: requiredText("Issuer", 500),
  credentialId: optionalText(500),
  credentialUrl: optionalHttpUrl("Credential URL"),
  description: optionalText(10_000),
  issueYear: yearSchema,
  issueMonth: monthSchema,
  expiryYear: yearSchema,
  expiryMonth: monthSchema,
  doesNotExpire: z.preprocess(booleanFromForm, z.boolean().default(false)),
  featured: featuredSchema,
  status: statusSchema,
  sortOrder: sortOrderSchema,
};

const nestedItemIdSchema = idSchema.optional();
const nestedRevisionSchema = z.iso.datetime({
  offset: true,
  error: "Refresh this record before saving nested content.",
});
const nestedRemovalShape = {
  revision: nestedRevisionSchema,
  removedIds: z.array(idSchema).max(300),
  confirmRemoval: z.literal(true).optional(),
};

function validateUniqueNestedProperty(
  items: readonly Record<string, unknown>[],
  property: "id" | "key" | "slug",
  section: string,
  label: string,
  context: z.RefinementCtx,
) {
  const firstIndexByValue = new Map<string, number>();
  items.forEach((item, index) => {
    const value = item[property];
    if (typeof value !== "string") return;
    if (firstIndexByValue.has(value)) {
      context.addIssue({
        code: "custom",
        path: [section, index, property],
        message: `${label} values must be unique within this list.`,
      });
      return;
    }
    firstIndexByValue.set(value, index);
  });
}

function validateUniqueRemovedIds(
  removedIds: readonly string[],
  context: z.RefinementCtx,
) {
  const seen = new Set<string>();
  removedIds.forEach((id, index) => {
    if (seen.has(id)) {
      context.addIssue({
        code: "custom",
        path: ["removedIds", index],
        message: "Removed record identifiers must be unique.",
      });
    }
    seen.add(id);
  });
}

const researchQuestionInputSchema = z
  .object({
    id: nestedItemIdSchema,
    key: keySchema,
    question: requiredText("Question", 2_000),
  })
  .strict();

const projectContributionInputSchema = z
  .object({
    id: nestedItemIdSchema,
    key: keySchema,
    label: requiredText("Label", 240),
    description: requiredText("Description", 10_000),
  })
  .strict();

const projectFeatureInputSchema = z
  .object({
    id: nestedItemIdSchema,
    key: keySchema,
    name: requiredText("Name", 240),
    description: optionalText(10_000),
  })
  .strict();

const projectTechnologyInputSchema = z
  .object({
    id: nestedItemIdSchema,
    slug: slugSchema,
    name: requiredText("Name", 240),
  })
  .strict();

export const researchNestedContentInputSchema = z
  .object({
    ...nestedRemovalShape,
    questions: z.array(researchQuestionInputSchema).max(25),
  })
  .strict()
  .superRefine((data, context) => {
    validateUniqueNestedProperty(
      data.questions,
      "id",
      "questions",
      "Question identifiers",
      context,
    );
    validateUniqueNestedProperty(
      data.questions,
      "key",
      "questions",
      "Question keys",
      context,
    );
    validateUniqueRemovedIds(data.removedIds, context);
  });

export const projectNestedContentInputSchema = z
  .object({
    ...nestedRemovalShape,
    contributions: z.array(projectContributionInputSchema).max(30),
    features: z.array(projectFeatureInputSchema).max(40),
    technologies: z.array(projectTechnologyInputSchema).max(50),
  })
  .strict()
  .superRefine((data, context) => {
    const sections = [
      ["contributions", data.contributions, "key", "Contribution keys"],
      ["features", data.features, "key", "Feature keys"],
      ["technologies", data.technologies, "slug", "Technology slugs"],
    ] as const;
    const firstPathById = new Map<string, readonly [string, number]>();

    for (const [section, items, naturalKey, label] of sections) {
      validateUniqueNestedProperty(items, naturalKey, section, label, context);
      items.forEach((item, index) => {
        if (!item.id) return;
        if (firstPathById.has(item.id)) {
          context.addIssue({
            code: "custom",
            path: [section, index, "id"],
            message: "Nested record identifiers must be unique.",
          });
          return;
        }
        firstPathById.set(item.id, [section, index]);
      });
    }
    validateUniqueRemovedIds(data.removedIds, context);
  });

export const adminResourceSchema = z.enum(adminResourceKeys);
export const adminNestedResourceSchema = z.enum(adminNestedResourceKeys);
export const adminRecordIdSchema = idSchema;
export const adminVisibilitySchema = z.enum(EDITABLE_CONTENT_STATUSES);
export const adminDeleteInputSchema = z
  .object({ confirmed: z.literal(true) })
  .strict();

export const adminCreateInputSchemas = {
  profile: z.object(profileShape).strict(),
  "social-links": z
    .object(socialLinkShape)
    .strict()
    .superRefine(validateSocialLinkProtocol),
  "research-interests": z.object(researchInterestShape).strict(),
  "research-projects": z
    .object(researchProjectShape)
    .strict()
    .superRefine((data, context) => validatePeriod(data, context, false)),
  projects: z
    .object(projectShape)
    .strict()
    .superRefine((data, context) => validatePeriod(data, context, false)),
  experience: z
    .object(experienceShape)
    .strict()
    .superRefine((data, context) => validatePeriod(data, context, false)),
  education: z
    .object(educationShape)
    .strict()
    .superRefine((data, context) => validateEducation(data, context, false)),
  publications: z.object(publicationShape).strict(),
  articles: z.object(articleShape).strict(),
  resumes: z.object(resumeShape).strict(),
  "skill-categories": z.object(skillCategoryShape).strict(),
  skills: z.object(skillShape).strict(),
  certifications: z
    .object(certificationShape)
    .strict()
    .superRefine((data, context) =>
      validateCertification(data, context, false),
    ),
  "site-settings": z.never({
    error: "Site settings are read-only until a key is explicitly allowlisted.",
  }),
} satisfies Record<AdminResourceKey, z.ZodType>;

// Edit forms submit a complete record. Reusing the create schemas ensures a
// forged partial Server Action payload cannot bypass a cross-field invariant
// that depends on a value already stored in PostgreSQL.
export const adminUpdateInputSchemas = adminCreateInputSchemas;

export type AdminCreateInputMap = {
  [Resource in AdminResourceKey]: z.output<
    (typeof adminCreateInputSchemas)[Resource]
  >;
};

export type AdminUpdateInputMap = {
  [Resource in AdminResourceKey]: z.output<
    (typeof adminUpdateInputSchemas)[Resource]
  >;
};

export const adminNestedContentInputSchemas = {
  "research-projects": researchNestedContentInputSchema,
  projects: projectNestedContentInputSchema,
} satisfies {
  [Resource in AdminNestedResourceKey]: z.ZodType<
    AdminNestedContentInputMap[Resource]
  >;
};

export function parseAdminCreateInput<Resource extends AdminResourceKey>(
  resource: Resource,
  input: unknown,
): AdminCreateInputMap[Resource] {
  return adminCreateInputSchemas[resource].parse(
    input,
  ) as AdminCreateInputMap[Resource];
}

export function parseAdminUpdateInput<Resource extends AdminResourceKey>(
  resource: Resource,
  input: unknown,
): AdminUpdateInputMap[Resource] {
  return adminUpdateInputSchemas[resource].parse(
    input,
  ) as AdminUpdateInputMap[Resource];
}

export function parseAdminNestedContentInput<
  Resource extends AdminNestedResourceKey,
>(
  resource: Resource,
  input: unknown,
): AdminNestedContentInputMap[Resource] {
  return adminNestedContentInputSchemas[resource].parse(
    input,
  ) as AdminNestedContentInputMap[Resource];
}
