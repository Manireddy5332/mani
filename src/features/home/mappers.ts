import { formatPartialPeriod } from "@/lib/public-content/dates";

import type { HomePageData } from "./types";

type Period = Readonly<{
  startYear: number | null;
  startMonth: number | null;
  endYear: number | null;
  endMonth: number | null;
  isCurrent: boolean;
}>;

export type PublicHomeRecord = Readonly<{
  name: string;
  positioning: readonly string[];
  introduction: string;
  location: string | null;
  socialLinks: readonly Readonly<{
    kind: string;
    url: string;
  }>[];
  experiences: readonly (Period &
    Readonly<{
      type: string;
      organization: string;
      location: string | null;
      role: string;
      summary: string;
    }>)[];
  educationRecords: readonly (Period &
    Readonly<{
      degree: string;
      institution: string;
    }>)[];
  researchInterests: readonly Readonly<{ name: string }>[];
  researchProjects: readonly Readonly<{
    slug: string;
    title: string;
    stage: string;
    format: string | null;
    summary: string;
    questions: readonly Readonly<{ question: string }>[];
  }>[];
  projects: readonly Readonly<{
    slug: string;
    title: string;
    type: string | null;
    institution: string | null;
    advisor: string | null;
    role: string | null;
    summary: string;
    repositoryUrl: string | null;
    technologies: readonly Readonly<{ name: string }>[];
  }>[];
  skillCategories: readonly Readonly<{
    name: string;
    skills: readonly Readonly<{ name: string }>[];
  }>[];
}>;

function enumLabel(value: string): string {
  const normalized = value.replaceAll("_", " ").toLowerCase();
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function safeLink(value: string | null, protocol: "http" | "mailto") {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (protocol === "mailto") {
      return url.protocol === "mailto:" && url.pathname.trim() ? value : null;
    }
    return (url.protocol === "https:" || url.protocol === "http:") &&
      !url.username &&
      !url.password
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

export function mapPublicHomeRecord(record: PublicHomeRecord): HomePageData {
  const email = record.socialLinks.find((link) => link.kind === "EMAIL");
  const linkedIn = record.socialLinks.find(
    (link) => link.kind === "LINKEDIN",
  );
  const research = record.researchProjects[0];
  const project = record.projects[0];

  return {
    identity: {
      name: record.name,
      positioning: record.positioning,
      introduction: record.introduction,
      location: record.location,
    },
    links: {
      email: safeLink(email?.url ?? null, "mailto"),
      linkedIn: safeLink(linkedIn?.url ?? null, "http"),
    },
    experience: record.experiences.map((item) => ({
      engagement: enumLabel(item.type),
      organization: item.organization,
      location: item.location,
      role: item.role,
      period: formatPartialPeriod(item),
      summary: item.summary,
    })),
    education: record.educationRecords.map((item) => ({
      degree: item.degree,
      institution: item.institution,
      period: formatPartialPeriod(item),
    })),
    researchInterests: record.researchInterests.map((item) => item.name),
    currentResearch: research
      ? {
          href: `/research/${research.slug}`,
          title: research.title,
          stage: enumLabel(research.stage),
          format: research.format,
          summary: research.summary,
          questions: research.questions.map((item) => item.question),
        }
      : null,
    selectedProject: project
      ? {
          href: `/projects/${project.slug}`,
          title: project.title,
          type: project.type,
          institution: project.institution,
          advisor: project.advisor,
          role: project.role,
          summary: project.summary,
          technologies: project.technologies.map((item) => item.name),
          repositoryUrl: safeLink(project.repositoryUrl, "http"),
        }
      : null,
    technicalExpertise: record.skillCategories
      .map((category) => ({
        category: category.name,
        skills: category.skills.map((skill) => skill.name),
      }))
      .filter((category) => category.skills.length > 0),
  };
}

