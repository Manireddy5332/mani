export const adminResourceKeys = [
  "profile",
  "social-links",
  "research-interests",
  "research-projects",
  "projects",
  "experience",
  "education",
  "publications",
  "articles",
  "resumes",
  "skill-categories",
  "skills",
  "certifications",
  "site-settings",
] as const;

export type AdminResourceKey = (typeof adminResourceKeys)[number];

export type AdminNavigationGroup =
  | "Identity"
  | "Research"
  | "Portfolio"
  | "Publishing"
  | "Configuration";

export type AdminFieldKind =
  | "checkbox"
  | "datetime"
  | "email"
  | "number"
  | "relation"
  | "select"
  | "slug"
  | "string-list"
  | "text"
  | "textarea"
  | "url";

export type AdminFieldValue = string | number | boolean | string[] | null;

export type AdminFieldOption = {
  label: string;
  value: string;
};

export type AdminFieldDefinition = {
  name: string;
  label: string;
  kind: AdminFieldKind;
  section: string;
  description?: string;
  placeholder?: string;
  required?: boolean;
  options?: readonly AdminFieldOption[];
  optionsResource?: AdminResourceKey;
  min?: number;
  max?: number;
  step?: number;
  rows?: number;
  autocomplete?: string;
  fullWidth?: boolean;
};

export type AdminResourceCapabilities = {
  create: boolean;
  update: boolean;
  delete: boolean;
};

export type AdminResourceDefinition = {
  key: AdminResourceKey;
  label: string;
  singularLabel: string;
  description: string;
  emptyTitle: string;
  emptyDescription: string;
  navigationGroup: AdminNavigationGroup;
  icon:
    | "article"
    | "award"
    | "briefcase"
    | "education"
    | "folder"
    | "link"
    | "profile"
    | "publication"
    | "research"
    | "resume"
    | "settings"
    | "skill";
  capabilities: AdminResourceCapabilities;
  fields: readonly AdminFieldDefinition[];
};

export type AdminRecordSummaryDto = {
  id: string;
  title: string;
  subtitle?: string;
  status?: string;
  featured?: boolean;
  sortOrder?: number;
  updatedAt: string;
};

export type AdminRecordDto = AdminRecordSummaryDto & {
  values: Record<string, AdminFieldValue>;
};

export type AdminDashboardResourceDto = {
  resource: AdminResourceKey;
  count: number;
  draftCount: number;
  publishedCount: number;
};

export type AdminDashboardDto = {
  profileExists: boolean;
  totalRecords: number;
  resources: AdminDashboardResourceDto[];
};

export type AdminActionResult = {
  ok: boolean;
  message: string;
  record?: AdminRecordDto;
  fieldErrors?: Record<string, string[]>;
};

export const adminNestedResourceKeys = [
  "research-projects",
  "projects",
] as const;

export type AdminNestedResourceKey =
  (typeof adminNestedResourceKeys)[number];

export type AdminVisibilityStatus = "DRAFT" | "PUBLISHED";

export type AdminNestedParentDto = {
  id: string;
  title: string;
  status: "ARCHIVED" | AdminVisibilityStatus;
  source: "ACADEMIC_CV" | "USER_PROVIDED" | "ADMIN";
  publishedAt: string | null;
  updatedAt: string;
};

export type AdminResearchQuestionDto = {
  id: string;
  key: string;
  question: string;
  sortOrder: number;
};

export type AdminProjectContributionDto = {
  id: string;
  key: string;
  label: string;
  description: string;
  sortOrder: number;
};

export type AdminProjectFeatureDto = {
  id: string;
  key: string;
  name: string;
  description: string | null;
  sortOrder: number;
};

export type AdminProjectTechnologyDto = {
  id: string;
  slug: string;
  name: string;
  sortOrder: number;
};

export type AdminResearchNestedContentDto = {
  resource: "research-projects";
  parent: AdminNestedParentDto;
  questions: AdminResearchQuestionDto[];
};

export type AdminProjectNestedContentDto = {
  resource: "projects";
  parent: AdminNestedParentDto;
  contributions: AdminProjectContributionDto[];
  features: AdminProjectFeatureDto[];
  technologies: AdminProjectTechnologyDto[];
};

export type AdminNestedContentDto =
  | AdminResearchNestedContentDto
  | AdminProjectNestedContentDto;

export type AdminResearchQuestionInput = {
  id?: string;
  key: string;
  question: string;
};

export type AdminProjectContributionInput = {
  id?: string;
  key: string;
  label: string;
  description: string;
};

export type AdminProjectFeatureInput = {
  id?: string;
  key: string;
  name: string;
  description?: string | null;
};

export type AdminProjectTechnologyInput = {
  id?: string;
  slug: string;
  name: string;
};

type AdminNestedWriteConfirmation = {
  revision: string;
  removedIds: string[];
  confirmRemoval?: true;
};

export type AdminResearchNestedContentInput = AdminNestedWriteConfirmation & {
  questions: AdminResearchQuestionInput[];
};

export type AdminProjectNestedContentInput = AdminNestedWriteConfirmation & {
  contributions: AdminProjectContributionInput[];
  features: AdminProjectFeatureInput[];
  technologies: AdminProjectTechnologyInput[];
};

export type AdminNestedContentInputMap = {
  "research-projects": AdminResearchNestedContentInput;
  projects: AdminProjectNestedContentInput;
};

export type AdminNestedRemovalConfirmation = {
  removedIds: string[];
  labels: string[];
};

export type AdminNestedActionResult = {
  ok: boolean;
  message: string;
  content?: AdminNestedContentDto;
  fieldErrors?: Record<string, string[]>;
  confirmation?: AdminNestedRemovalConfirmation;
};

export type AdminFormMode = "create" | "edit";

export type AdminRelationOptions = Partial<
  Record<AdminResourceKey, AdminRecordSummaryDto[]>
>;
