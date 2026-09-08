import {
  adminResourceKeys,
  type AdminFieldDefinition,
  type AdminNavigationGroup,
  type AdminResourceDefinition,
  type AdminResourceKey,
} from "./types";

const contentStatusOptions = [
  { label: "Draft", value: "DRAFT" },
  { label: "Published", value: "PUBLISHED" },
] as const;

const socialLinkKindOptions = [
  { label: "Email", value: "EMAIL" },
  { label: "LinkedIn", value: "LINKEDIN" },
  { label: "GitHub", value: "GITHUB" },
  { label: "Website", value: "WEBSITE" },
  { label: "Other", value: "OTHER" },
] as const;

const researchStageOptions = [
  { label: "Research interest", value: "RESEARCH_INTEREST" },
  { label: "Exploring", value: "EXPLORING" },
  { label: "Early stage", value: "EARLY_STAGE" },
  { label: "In progress", value: "IN_PROGRESS" },
  { label: "Working paper", value: "WORKING_PAPER" },
  { label: "Submitted", value: "SUBMITTED" },
  { label: "Accepted", value: "ACCEPTED" },
  { label: "Published", value: "PUBLISHED" },
  { label: "Archived", value: "ARCHIVED" },
] as const;

const publicationStageOptions = [
  { label: "Working paper", value: "WORKING_PAPER" },
  { label: "Submitted", value: "SUBMITTED" },
  { label: "Accepted", value: "ACCEPTED" },
  { label: "Published", value: "PUBLISHED" },
  { label: "Archived", value: "ARCHIVED" },
] as const;

const publicationTypeOptions = [
  { label: "Journal article", value: "JOURNAL_ARTICLE" },
  { label: "Conference paper", value: "CONFERENCE_PAPER" },
  { label: "Workshop paper", value: "WORKSHOP_PAPER" },
  { label: "Preprint", value: "PREPRINT" },
  { label: "Book chapter", value: "BOOK_CHAPTER" },
  { label: "Thesis", value: "THESIS" },
  { label: "Other", value: "OTHER" },
] as const;

const projectStatusOptions = [
  { label: "Planned", value: "PLANNED" },
  { label: "In progress", value: "IN_PROGRESS" },
  { label: "On hold", value: "ON_HOLD" },
  { label: "Completed", value: "COMPLETED" },
  { label: "Archived", value: "ARCHIVED" },
] as const;

const experienceTypeOptions = [
  { label: "Client engagement", value: "CLIENT_ENGAGEMENT" },
  { label: "Employment", value: "EMPLOYMENT" },
  { label: "Internship", value: "INTERNSHIP" },
  { label: "Contract", value: "CONTRACT" },
  { label: "Other", value: "OTHER" },
] as const;

function field(
  name: string,
  label: string,
  kind: AdminFieldDefinition["kind"],
  section: string,
  options: Omit<AdminFieldDefinition, "name" | "label" | "kind" | "section"> = {},
): AdminFieldDefinition {
  return { name, label, kind, section, ...options };
}

const statusField = field("status", "Visibility", "select", "Publishing", {
  required: true,
  options: contentStatusOptions,
  description: "Draft records remain private. Published records become public after you save.",
});

const featuredField = field("featured", "Featured", "checkbox", "Publishing", {
  description: "Mark this record for featured placement where the public design supports it.",
});

const sortOrderField = field("sortOrder", "Sort order", "number", "Publishing", {
  min: 0,
  step: 1,
  description: "Lower numbers appear first when records share the same visibility state.",
});

function periodFields(section = "Timeline"): readonly AdminFieldDefinition[] {
  return [
    field("startYear", "Start year", "number", section, { min: 1900, max: 2200, step: 1 }),
    field("startMonth", "Start month", "number", section, { min: 1, max: 12, step: 1 }),
    field("endYear", "End year", "number", section, { min: 1900, max: 2200, step: 1 }),
    field("endMonth", "End month", "number", section, { min: 1, max: 12, step: 1 }),
    field("isCurrent", "Current", "checkbox", section, {
      description: "Current records must not include an end month or year.",
    }),
  ];
}

const registry = {
  profile: {
    key: "profile",
    label: "Profile & about",
    singularLabel: "profile",
    description: "Manage the primary identity, positioning, introduction, and about narrative.",
    emptyTitle: "No database profile yet",
    emptyDescription: "Create the primary profile before adding profile-linked portfolio records. Nothing is filled automatically.",
    navigationGroup: "Identity",
    icon: "profile",
    capabilities: { create: true, update: true, delete: false },
    fields: [
      field("name", "Name", "text", "Identity", { required: true, autocomplete: "name" }),
      field("slug", "Public slug", "slug", "Identity", {
        required: true,
        placeholder: "manikanta-reddy-anugu",
        description: "Use lowercase words separated by hyphens.",
      }),
      field("positioning", "Positioning", "string-list", "Identity", {
        description: "Enter one verified role or positioning statement per line.",
        fullWidth: true,
      }),
      field("headline", "Headline", "textarea", "Narrative", { required: true, rows: 3, fullWidth: true }),
      field("introduction", "Introduction", "textarea", "Narrative", { required: true, rows: 5, fullWidth: true }),
      field("about", "About paragraphs", "string-list", "Narrative", {
        description: "Enter one paragraph per line. Keep every statement grounded in verified information.",
        rows: 8,
        fullWidth: true,
      }),
      field("location", "Location", "text", "Identity", { autocomplete: "address-level2" }),
      statusField,
    ],
  },
  "social-links": {
    key: "social-links",
    label: "Social links",
    singularLabel: "social link",
    description: "Manage verified contact and professional profile destinations.",
    emptyTitle: "No social links",
    emptyDescription: "Add only links and contact methods that have been verified for public use.",
    navigationGroup: "Identity",
    icon: "link",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("key", "Internal key", "slug", "Link", { required: true, placeholder: "linkedin" }),
      field("kind", "Link type", "select", "Link", { required: true, options: socialLinkKindOptions }),
      field("label", "Label", "text", "Link", { required: true }),
      field("url", "Destination", "url", "Link", { required: true, fullWidth: true }),
      field("handle", "Handle", "text", "Link"),
      sortOrderField,
      statusField,
    ],
  },
  "research-interests": {
    key: "research-interests",
    label: "Research interests",
    singularLabel: "research interest",
    description: "Maintain the verified themes that frame the research portfolio.",
    emptyTitle: "No research interests",
    emptyDescription: "Add a research interest only when it reflects a verified current direction.",
    navigationGroup: "Research",
    icon: "research",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("name", "Name", "text", "Interest", { required: true }),
      field("slug", "Public slug", "slug", "Interest", { required: true }),
      field("description", "Description", "textarea", "Interest", { rows: 4, fullWidth: true }),
      sortOrderField,
      statusField,
    ],
  },
  "research-projects": {
    key: "research-projects",
    label: "Research projects",
    singularLabel: "research project",
    description: "Manage research directions with explicit stages, scope, methods, and evidence boundaries.",
    emptyTitle: "No research projects",
    emptyDescription: "The management structure is ready; leave it empty until a direction is verified.",
    navigationGroup: "Research",
    icon: "research",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("title", "Title", "text", "Overview", { required: true, fullWidth: true }),
      field("slug", "Public slug", "slug", "Overview", { required: true }),
      field("summary", "Summary", "textarea", "Overview", { required: true, rows: 5, fullWidth: true }),
      field("abstract", "Abstract", "textarea", "Narrative", { rows: 7, fullWidth: true }),
      field("motivation", "Motivation", "textarea", "Narrative", { rows: 5, fullWidth: true }),
      field("methodology", "Methodology", "textarea", "Methods & evidence", { rows: 5, fullWidth: true }),
      field("methodologySummary", "Methodology summary", "textarea", "Methods & evidence", { rows: 4, fullWidth: true }),
      field("scopeBoundary", "Scope boundary", "textarea", "Methods & evidence", { rows: 4, fullWidth: true }),
      field("evidenceStatus", "Evidence status", "textarea", "Methods & evidence", { rows: 4, fullWidth: true }),
      field("domain", "Domain", "text", "Classification"),
      field("researchArea", "Research area", "text", "Classification"),
      field("format", "Format", "text", "Classification"),
      field("advisor", "Advisor", "text", "People"),
      field("collaborators", "Collaborators", "string-list", "People", { description: "One verified collaborator per line.", fullWidth: true }),
      field("technologies", "Technologies", "string-list", "Methods & evidence", { fullWidth: true }),
      field("datasets", "Datasets", "string-list", "Methods & evidence", { fullWidth: true }),
      field("githubUrl", "Repository URL", "url", "Links"),
      field("paperUrl", "Paper URL", "url", "Links"),
      field("progressPercent", "Progress", "number", "Timeline", { min: 0, max: 100, step: 1 }),
      field("stage", "Research stage", "select", "Timeline", { required: true, options: researchStageOptions }),
      ...periodFields(),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  projects: {
    key: "projects",
    label: "Projects",
    singularLabel: "project",
    description: "Manage verified academic and professional project case-study records.",
    emptyTitle: "No projects",
    emptyDescription: "Add a project only when its role, scope, and supporting details are verified.",
    navigationGroup: "Portfolio",
    icon: "folder",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("title", "Title", "text", "Overview", { required: true, fullWidth: true }),
      field("shortTitle", "Short title", "text", "Overview"),
      field("slug", "Public slug", "slug", "Overview", { required: true }),
      field("type", "Type", "text", "Classification"),
      field("category", "Category", "text", "Classification"),
      field("summary", "Summary", "textarea", "Overview", { required: true, rows: 5, fullWidth: true }),
      field("description", "Description", "textarea", "Narrative", { rows: 7, fullWidth: true }),
      field("problemStatement", "Problem statement", "textarea", "Narrative", { rows: 5, fullWidth: true }),
      field("motivation", "Motivation", "textarea", "Narrative", { rows: 5, fullWidth: true }),
      field("approach", "Approach", "textarea", "Implementation", { rows: 6, fullWidth: true }),
      field("architecture", "Architecture", "textarea", "Implementation", { rows: 6, fullWidth: true }),
      field("implementation", "Implementation", "text", "Implementation", { fullWidth: true }),
      field("aiModels", "AI models", "string-list", "Implementation", { fullWidth: true }),
      field("datasets", "Datasets", "string-list", "Implementation", { fullWidth: true }),
      field("challenges", "Challenges", "string-list", "Evidence", { fullWidth: true }),
      field("results", "Results", "string-list", "Evidence", { description: "Only enter results supported by verified evidence.", fullWidth: true }),
      field("lessons", "Lessons", "string-list", "Evidence", { fullWidth: true }),
      field("institution", "Institution", "text", "Context"),
      field("advisor", "Advisor", "text", "Context"),
      field("role", "Role", "text", "Context"),
      field("repositoryUrl", "Repository URL", "url", "Links"),
      field("demoUrl", "Demo URL", "url", "Links"),
      field("documentationUrl", "Documentation URL", "url", "Links"),
      field("seoTitle", "SEO title", "text", "Search metadata", { fullWidth: true }),
      field("seoDescription", "SEO description", "textarea", "Search metadata", { rows: 3, fullWidth: true }),
      field("projectStatus", "Project status", "select", "Timeline", { options: projectStatusOptions }),
      ...periodFields(),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  experience: {
    key: "experience",
    label: "Experience",
    singularLabel: "experience record",
    description: "Manage verified employment, client engagement, contract, and internship history.",
    emptyTitle: "No experience records",
    emptyDescription: "Add only roles, organizations, dates, and details supported by the Academic CV or later verified information.",
    navigationGroup: "Portfolio",
    icon: "briefcase",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("organization", "Organization", "text", "Role", { required: true }),
      field("role", "Role", "text", "Role", { required: true }),
      field("slug", "Public slug", "slug", "Role", { required: true }),
      field("type", "Experience type", "select", "Role", { required: true, options: experienceTypeOptions }),
      field("location", "Location", "text", "Role"),
      field("domain", "Domain", "text", "Role"),
      field("summary", "Summary", "textarea", "Details", { required: true, rows: 5, fullWidth: true }),
      field("practiceAreas", "Practice areas", "string-list", "Details", { fullWidth: true }),
      field("highlights", "Highlights", "string-list", "Details", { description: "Only enter evidence-backed highlights; omit confidential metrics.", fullWidth: true }),
      ...periodFields(),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  education: {
    key: "education",
    label: "Education",
    singularLabel: "education record",
    description: "Manage verified degrees and academic history without inferring unsupported details.",
    emptyTitle: "No education records",
    emptyDescription: "Add only degrees, institutions, dates, and details supported by verified sources.",
    navigationGroup: "Portfolio",
    icon: "education",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("institution", "Institution", "text", "Program", { required: true, fullWidth: true }),
      field("degree", "Degree", "text", "Program", { required: true, fullWidth: true }),
      field("field", "Field", "text", "Program"),
      field("slug", "Public slug", "slug", "Program", { required: true }),
      field("location", "Location", "text", "Program"),
      ...periodFields(),
      field("gpa", "GPA", "number", "Academic details", { min: 0, step: 0.01 }),
      field("gpaScale", "GPA scale", "number", "Academic details", { min: 0.01, step: 0.01 }),
      field("coursework", "Coursework", "string-list", "Academic details", { fullWidth: true }),
      field("thesis", "Thesis", "textarea", "Academic details", { rows: 4, fullWidth: true }),
      field("activities", "Activities", "string-list", "Academic details", { fullWidth: true }),
      field("url", "Institution or program URL", "url", "Program"),
      sortOrderField,
      statusField,
    ],
  },
  publications: {
    key: "publications",
    label: "Publications",
    singularLabel: "publication",
    description: "Manage verified scholarly outputs with accurate authorship, stage, venue, and identifiers.",
    emptyTitle: "No verified publications",
    emptyDescription: "This section is intentionally empty until a publication is verified. The absence of records is not an error.",
    navigationGroup: "Research",
    icon: "publication",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("title", "Title", "text", "Citation", { required: true, fullWidth: true }),
      field("slug", "Public slug", "slug", "Citation", { required: true }),
      field("authors", "Authors", "string-list", "Citation", { required: true, description: "One author per line, in publication order.", fullWidth: true }),
      field("venue", "Venue", "text", "Citation"),
      field("type", "Publication type", "select", "Citation", { required: true, options: publicationTypeOptions }),
      field("year", "Year", "number", "Citation", { min: 1900, max: 2200, step: 1 }),
      field("stage", "Publication stage", "select", "Citation", { required: true, options: publicationStageOptions }),
      field("abstract", "Abstract", "textarea", "Details", { rows: 8, fullWidth: true }),
      field("doi", "DOI", "text", "Identifiers"),
      field("arxivUrl", "arXiv URL", "url", "Identifiers"),
      field("paperUrl", "Paper URL", "url", "Identifiers"),
      field("citation", "Formatted citation", "textarea", "Citation", { rows: 4, fullWidth: true }),
      field("bibtex", "BibTeX", "textarea", "Citation", { rows: 7, fullWidth: true }),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  articles: {
    key: "articles",
    label: "Articles & writing",
    singularLabel: "article",
    description: "Prepare draft or published research notes and long-form writing records.",
    emptyTitle: "No writing entries",
    emptyDescription: "The writing archive can remain empty until a verified article or research note is ready.",
    navigationGroup: "Publishing",
    icon: "article",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("title", "Title", "text", "Article", { required: true, fullWidth: true }),
      field("slug", "Public slug", "slug", "Article", { required: true }),
      field("category", "Category", "text", "Article"),
      field("excerpt", "Excerpt", "textarea", "Article", { rows: 4, fullWidth: true }),
      field("content", "Content", "textarea", "Content", { required: true, rows: 18, fullWidth: true, description: "Store reviewed Markdown text here. Rich formatting is intentionally kept out of this focused editor." }),
      field("contentFormat", "Content format", "select", "Content", {
        required: true,
        options: [{ label: "Markdown", value: "markdown" }],
      }),
      field("estimatedReadingMinutes", "Estimated reading time", "number", "Article", { min: 1, step: 1 }),
      field("seoTitle", "SEO title", "text", "Search metadata", { fullWidth: true }),
      field("seoDescription", "SEO description", "textarea", "Search metadata", { rows: 3, fullWidth: true }),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  resumes: {
    key: "resumes",
    label: "Resume & CV",
    singularLabel: "resume record",
    description: "Manage resume metadata while file storage remains provider-neutral and unconfigured.",
    emptyTitle: "No resume records",
    emptyDescription: "Create metadata only when a verified resume version is ready. File upload is intentionally deferred.",
    navigationGroup: "Identity",
    icon: "resume",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("title", "Title", "text", "Resume", { required: true, fullWidth: true }),
      field("slug", "Public slug", "slug", "Resume", { required: true }),
      field("version", "Version", "text", "Resume"),
      field("isCurrent", "Current resume", "checkbox", "Resume", { description: "Only one current resume is allowed for the profile." }),
      statusField,
    ],
  },
  "skill-categories": {
    key: "skill-categories",
    label: "Skill categories",
    singularLabel: "skill category",
    description: "Organize verified technical expertise into ordered groups.",
    emptyTitle: "No skill categories",
    emptyDescription: "Create a category before adding individual skills.",
    navigationGroup: "Portfolio",
    icon: "skill",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("name", "Name", "text", "Category", { required: true }),
      field("slug", "Slug", "slug", "Category", { required: true }),
      field("description", "Description", "textarea", "Category", { rows: 4, fullWidth: true }),
      sortOrderField,
      statusField,
    ],
  },
  skills: {
    key: "skills",
    label: "Skills",
    singularLabel: "skill",
    description: "Manage individual verified skills within an existing category.",
    emptyTitle: "No skills",
    emptyDescription: "Add verified skills after creating at least one skill category.",
    navigationGroup: "Portfolio",
    icon: "skill",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("categoryId", "Skill category", "relation", "Skill", { required: true, optionsResource: "skill-categories" }),
      field("name", "Name", "text", "Skill", { required: true }),
      field("slug", "Slug", "slug", "Skill", { required: true }),
      field("description", "Description", "textarea", "Skill", { rows: 4, fullWidth: true }),
      sortOrderField,
      statusField,
    ],
  },
  certifications: {
    key: "certifications",
    label: "Certifications",
    singularLabel: "certification",
    description: "Manage verified certifications without creating unsupported credentials or dates.",
    emptyTitle: "No verified certifications",
    emptyDescription: "This section is intentionally empty until a certification is supported by verified information.",
    navigationGroup: "Portfolio",
    icon: "award",
    capabilities: { create: true, update: true, delete: true },
    fields: [
      field("name", "Certification name", "text", "Credential", { required: true, fullWidth: true }),
      field("issuer", "Issuer", "text", "Credential", { required: true }),
      field("slug", "Public slug", "slug", "Credential", { required: true }),
      field("credentialId", "Credential ID", "text", "Credential"),
      field("credentialUrl", "Credential URL", "url", "Credential"),
      field("description", "Description", "textarea", "Credential", { rows: 5, fullWidth: true }),
      field("issueYear", "Issue year", "number", "Dates", { min: 1900, max: 2200, step: 1 }),
      field("issueMonth", "Issue month", "number", "Dates", { min: 1, max: 12, step: 1 }),
      field("expiryYear", "Expiry year", "number", "Dates", { min: 1900, max: 2200, step: 1 }),
      field("expiryMonth", "Expiry month", "number", "Dates", { min: 1, max: 12, step: 1 }),
      field("doesNotExpire", "Does not expire", "checkbox", "Dates"),
      featuredField,
      sortOrderField,
      statusField,
    ],
  },
  "site-settings": {
    key: "site-settings",
    label: "Site settings",
    singularLabel: "site setting",
    description: "Reserved for explicitly defined, validated site-level configuration.",
    emptyTitle: "No editable site settings",
    emptyDescription: "Settings appear only after a key has a dedicated validator and purpose-built control; unrestricted JSON editing remains disabled.",
    navigationGroup: "Configuration",
    icon: "settings",
    capabilities: { create: false, update: false, delete: false },
    fields: [],
  },
} satisfies Record<AdminResourceKey, AdminResourceDefinition>;

export const adminResources = adminResourceKeys.map((key) => registry[key]);

export const adminNavigationGroups: readonly AdminNavigationGroup[] = [
  "Identity",
  "Research",
  "Portfolio",
  "Publishing",
  "Configuration",
];

export function isAdminResourceKey(value: string): value is AdminResourceKey {
  return (adminResourceKeys as readonly string[]).includes(value);
}

export function getAdminResource(value: string): AdminResourceDefinition | null {
  return isAdminResourceKey(value) ? registry[value] : null;
}

export function getAdminResourcePath(resource: AdminResourceKey): `/admin/${AdminResourceKey}` {
  return `/admin/${resource}`;
}

export function getAdminRelationResources(
  resource: AdminResourceDefinition,
): AdminResourceKey[] {
  return Array.from(
    new Set(
      resource.fields.flatMap((item) =>
        item.optionsResource ? [item.optionsResource] : [],
      ),
    ),
  );
}
