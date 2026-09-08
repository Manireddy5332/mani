import {
  formatMonthYear,
  formatPartialPeriod,
} from "@/lib/public-content/dates";

import type {
  ProfileCertification,
  ProfilePageData,
  ResumePageData,
  ResumeProject,
} from "./types";

type PartialPeriodRecord = Readonly<{
  startYear: number | null;
  startMonth: number | null;
  endYear: number | null;
  endMonth: number | null;
  isCurrent: boolean;
}>;

export type PublicProfileRecord = Readonly<{
  name: string;
  positioning: readonly string[];
  headline: string;
  introduction: string;
  about: readonly string[];
  location: string | null;
  experiences: readonly (PartialPeriodRecord &
    Readonly<{
      type: string;
      organization: string;
      role: string;
      location: string | null;
      summary: string;
    }>)[];
  educationRecords: readonly (PartialPeriodRecord &
    Readonly<{
      institution: string;
      degree: string;
    }>)[];
  researchInterests: readonly Readonly<{ name: string }>[];
  researchProjects: readonly Readonly<{
    slug: string;
    title: string;
    summary: string;
    format: string | null;
    stage: string;
    questions: readonly Readonly<{ question: string }>[];
  }>[];
  skillCategories: readonly Readonly<{
    name: string;
    skills: readonly Readonly<{ name: string }>[];
  }>[];
  certifications: readonly Readonly<{
    name: string;
    issuer: string;
    credentialUrl: string | null;
    issueYear: number | null;
    issueMonth: number | null;
    expiryYear: number | null;
    expiryMonth: number | null;
    doesNotExpire: boolean;
  }>[];
}>;

export type PublicSocialLinkRecord = Readonly<{
  kind: string;
  url: string;
}>;

export type PublicProjectRecord = Readonly<{
  slug: string;
  title: string;
  type: string | null;
  institution: string | null;
  advisor: string | null;
  role: string | null;
  summary: string;
  repositoryUrl: string | null;
  technologies: readonly Readonly<{ name: string }>[];
}>;

export type PublicResumeRecord = Readonly<{
  fileAsset: Readonly<{
    publicUrl: string | null;
    mimeType: string;
  }> | null;
}>;

export type PublicResumeProfileRecord = PublicProfileRecord &
  Readonly<{
    socialLinks: readonly PublicSocialLinkRecord[];
    projects: readonly PublicProjectRecord[];
    resumes: readonly PublicResumeRecord[];
  }>;

export const PUBLIC_RESUME_MIME_TYPES = [
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
] as const;

function formatEnumLabel(value: string): string {
  const normalized = value.replaceAll("_", " ").toLowerCase();
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

export function safeHttpUrl(value: string | null): string | null {
  if (!value) return null;

  try {
    const url = new URL(value);
    if (
      (url.protocol === "https:" || url.protocol === "http:") &&
      !url.username &&
      !url.password
    ) {
      return url.toString();
    }
  } catch {
    // An invalid optional URL is omitted from the public DTO.
  }

  return null;
}

export function safeMailtoUrl(value: string | null): string | null {
  if (!value) return null;

  try {
    const url = new URL(value);
    if (url.protocol === "mailto:" && url.pathname.trim()) {
      return value;
    }
  } catch {
    // An invalid optional URL is omitted from the public DTO.
  }

  return null;
}

export function emailAddressFromMailto(value: string | null): string | null {
  const mailto = safeMailtoUrl(value);
  if (!mailto) return null;

  try {
    return decodeURIComponent(new URL(mailto).pathname);
  } catch {
    return null;
  }
}

function mapCertification(
  record: PublicProfileRecord["certifications"][number],
): ProfileCertification {
  const issued = formatMonthYear(record.issueYear, record.issueMonth);
  const expiry = formatMonthYear(record.expiryYear, record.expiryMonth);
  const period = record.doesNotExpire
    ? issued
      ? `${issued} — Does not expire`
      : "Does not expire"
    : issued && expiry
      ? `${issued} — ${expiry}`
      : (issued ?? expiry);

  return {
    name: record.name,
    issuer: record.issuer,
    period,
    credentialUrl: safeHttpUrl(record.credentialUrl),
  };
}

export function mapPublicProfileRecord(
  record: PublicProfileRecord,
): ProfilePageData {
  const researchRecord = record.researchProjects[0];
  const researchInterests = record.researchInterests.map(
    (interest) => interest.name,
  );

  return {
    name: record.name,
    headline: record.headline,
    positioning: record.positioning,
    introduction: record.introduction,
    location: record.location,
    progression: record.about,
    experience: record.experiences.map((experience) => ({
      engagement: formatEnumLabel(experience.type),
      organization: experience.organization,
      location: experience.location,
      role: experience.role,
      period: formatPartialPeriod(experience),
      summary: experience.summary,
      isCurrent: experience.isCurrent,
    })),
    education: record.educationRecords.map((education) => ({
      degree: education.degree,
      institution: education.institution,
      period: formatPartialPeriod(education),
    })),
    research: researchRecord
      ? {
          title: researchRecord.title,
          href: `/research/${researchRecord.slug}`,
          stage: formatEnumLabel(researchRecord.stage),
          format: researchRecord.format,
          summary: researchRecord.summary,
          questions: researchRecord.questions.map(
            (question) => question.question,
          ),
          interests: researchInterests,
        }
      : null,
    expertise: record.skillCategories
      .map((category) => ({
        category: category.name,
        skills: category.skills.map((skill) => skill.name),
      }))
      .filter((category) => category.skills.length > 0),
    certifications: record.certifications.map(mapCertification),
  };
}

export function mapPublicProjectRecord(
  record: PublicProjectRecord,
): ResumeProject {
  return {
    href: `/projects/${record.slug}`,
    title: record.title,
    type: record.type,
    institution: record.institution,
    advisor: record.advisor,
    role: record.role,
    summary: record.summary,
    technologies: record.technologies.map((technology) => technology.name),
    repositoryUrl: safeHttpUrl(record.repositoryUrl),
  };
}

function eligibleResumeUrl(record: PublicResumeRecord | undefined) {
  if (
    !record?.fileAsset ||
    !PUBLIC_RESUME_MIME_TYPES.includes(
      record.fileAsset.mimeType as (typeof PUBLIC_RESUME_MIME_TYPES)[number],
    )
  ) {
    return null;
  }
  return safeHttpUrl(record.fileAsset.publicUrl);
}

export function mapPublicResumeRecord(
  record: PublicResumeProfileRecord,
): ResumePageData {
  const profile = mapPublicProfileRecord(record);
  const emailLink = record.socialLinks.find((link) => link.kind === "EMAIL");
  const linkedInLink = record.socialLinks.find(
    (link) => link.kind === "LINKEDIN",
  );
  const emailUrl = safeMailtoUrl(emailLink?.url ?? null);
  const emailAddress = emailAddressFromMailto(emailUrl);
  const cvRequestUrl = emailUrl
    ? `${emailUrl}${emailUrl.includes("?") ? "&" : "?"}subject=${encodeURIComponent(
        `Academic CV request for ${profile.name}`,
      )}`
    : null;

  return {
    ...profile,
    emailAddress,
    linkedInUrl: safeHttpUrl(linkedInLink?.url ?? null),
    cvRequestUrl,
    cvDownloadUrl: eligibleResumeUrl(record.resumes[0]),
    project: record.projects[0]
      ? mapPublicProjectRecord(record.projects[0])
      : null,
  };
}
