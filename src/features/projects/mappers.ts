import type {
  ProjectDetailRecord,
  ProjectPageRecord,
  ProjectsPageContent,
} from "./types";

export type PublicProjectContributionRow = Readonly<{
  description: string;
  id: string;
  label: string;
}>;

export type PublicProjectFeatureRow = Readonly<{
  id: string;
  name: string;
}>;

export type PublicProjectTechnologyRow = Readonly<{
  id: string;
  name: string;
}>;

export type PublicProjectRow = Readonly<{
  advisor: string | null;
  category: string | null;
  contributions: readonly PublicProjectContributionRow[];
  description: string | null;
  features: readonly PublicProjectFeatureRow[];
  id: string;
  implementation: string | null;
  institution: string | null;
  repositoryUrl: string | null;
  role: string | null;
  seoDescription: string | null;
  seoTitle: string | null;
  shortTitle: string | null;
  slug: string;
  summary: string;
  technologies: readonly PublicProjectTechnologyRow[];
  title: string;
  type: string | null;
}>;

function publicHttpUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return (url.protocol === "https:" || url.protocol === "http:") &&
      url.username === "" &&
      url.password === ""
      ? value
      : null;
  } catch {
    return null;
  }
}

export function mapProjectRecord(
  project: PublicProjectRow,
): ProjectDetailRecord {
  const shortTitle = project.shortTitle ?? project.title;
  const type = project.type ?? project.category ?? "Project";

  return {
    advisor: project.advisor,
    caseStudy: {
      contributionsDescription:
        project.contributions.length > 0
          ? "The public project record documents the following areas of responsibility."
          : "No public contribution records are listed yet.",
      contributionsTitle: "Documented responsibilities across the project.",
      learningDescription:
        project.features.length > 0
          ? "The public project record identifies the following features."
          : "No public feature records are listed yet.",
      learningTitle: "Features documented for this project.",
      overviewTitle: `Project overview: ${shortTitle}`,
      technologyDescription:
        project.technologies.length > 0
          ? "Technologies documented in the public project record."
          : "No public technology records are listed yet.",
      technologyTitle: "Documented technologies.",
    },
    contributions: project.contributions.map(({ description, label }) => ({
      description,
      label,
    })),
    href: `/projects/${project.slug}`,
    implementation: project.implementation,
    indexSummaryDescription: project.description ?? project.summary,
    indexSummaryTitle: shortTitle,
    institution: project.institution,
    learningFeatures: project.features.map((feature) => feature.name),
    repository: publicHttpUrl(project.repositoryUrl),
    role: project.role,
    seoDescription: project.seoDescription,
    seoTitle: project.seoTitle,
    shortTitle,
    slug: project.slug,
    summary: project.summary,
    technologies: project.technologies.map((technology) => technology.name),
    title: project.title,
    type,
  };
}

export function mapProjectsPageContent(
  projects: readonly PublicProjectRow[],
): ProjectsPageContent {
  return { projects: projects.map(mapProjectRecord) satisfies ProjectPageRecord[] };
}
