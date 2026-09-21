import type { PublicProfilePhoto } from "@/features/profile-photo/types";

export type ProfileExperience = {
  readonly engagement: string;
  readonly organization: string;
  readonly location: string | null;
  readonly role: string;
  readonly period: string | null;
  readonly summary: string;
  readonly isCurrent: boolean;
};

export type ProfileEducation = {
  readonly degree: string;
  readonly institution: string;
  readonly period: string | null;
};

export type SkillGroup = {
  readonly category: string;
  readonly skills: readonly string[];
};

export type ResearchDirection = {
  readonly title: string;
  readonly href: `/research/${string}`;
  readonly stage: string;
  readonly format: string | null;
  readonly summary: string;
  readonly questions: readonly string[];
  readonly interests: readonly string[];
};

export type ProfileCertification = {
  readonly name: string;
  readonly issuer: string;
  readonly period: string | null;
  readonly credentialUrl: string | null;
};

export type ProfilePageData = {
  readonly name: string;
  readonly headline: string;
  readonly positioning: readonly string[];
  readonly introduction: string;
  readonly location: string | null;
  readonly progression: readonly string[];
  readonly photo: PublicProfilePhoto | null;
  readonly experience: readonly ProfileExperience[];
  readonly education: readonly ProfileEducation[];
  readonly research: ResearchDirection | null;
  readonly expertise: readonly SkillGroup[];
  readonly certifications: readonly ProfileCertification[];
};

export type ResumeProject = {
  readonly href: `/projects/${string}`;
  readonly title: string;
  readonly type: string | null;
  readonly institution: string | null;
  readonly advisor: string | null;
  readonly role: string | null;
  readonly summary: string;
  readonly technologies: readonly string[];
  readonly repositoryUrl: string | null;
};

export type ResumePageData = ProfilePageData & {
  readonly emailAddress: string | null;
  readonly linkedInUrl: string | null;
  readonly cvRequestUrl: string | null;
  readonly cvDownloadUrl: string | null;
  readonly project: ResumeProject | null;
};
