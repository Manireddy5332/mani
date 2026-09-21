import type { PublicProfilePhoto } from "@/features/profile-photo/types";

export type HomeExperience = Readonly<{
  engagement: string;
  organization: string;
  location: string | null;
  role: string;
  period: string | null;
  summary: string;
}>;

export type HomeEducation = Readonly<{
  degree: string;
  institution: string;
  period: string | null;
}>;

export type HomeResearch = Readonly<{
  href: string;
  title: string;
  stage: string;
  format: string | null;
  summary: string;
  questions: readonly string[];
}>;

export type HomeProject = Readonly<{
  href: string;
  title: string;
  type: string | null;
  institution: string | null;
  advisor: string | null;
  role: string | null;
  summary: string;
  technologies: readonly string[];
  repositoryUrl: string | null;
}>;

export type HomePageData = Readonly<{
  identity: Readonly<{
    name: string;
    positioning: readonly string[];
    introduction: string;
    location: string | null;
    photo: PublicProfilePhoto | null;
  }>;
  links: Readonly<{
    email: string | null;
    linkedIn: string | null;
  }>;
  experience: readonly HomeExperience[];
  education: readonly HomeEducation[];
  researchInterests: readonly string[];
  currentResearch: HomeResearch | null;
  selectedProject: HomeProject | null;
  technicalExpertise: readonly Readonly<{
    category: string;
    skills: readonly string[];
  }>[];
}>;

