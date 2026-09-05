export type PublicExperienceType =
  | "CLIENT_ENGAGEMENT"
  | "EMPLOYMENT"
  | "INTERNSHIP"
  | "CONTRACT"
  | "OTHER";

export type ExperienceSourceRecord = {
  readonly type: PublicExperienceType;
  readonly location: string | null;
  readonly organization: string;
  readonly practiceAreas: readonly string[];
  readonly role: string;
  readonly summary: string;
  readonly startYear: number | null;
  readonly startMonth: number | null;
  readonly endYear: number | null;
  readonly endMonth: number | null;
  readonly isCurrent: boolean;
};

export type EducationSourceRecord = {
  readonly degree: string;
  readonly institution: string;
  readonly startYear: number | null;
  readonly startMonth: number | null;
  readonly endYear: number | null;
  readonly endMonth: number | null;
  readonly isCurrent: boolean;
};

export type SkillCategorySourceRecord = {
  readonly name: string;
  readonly skills: readonly {
    readonly name: string;
  }[];
};

export type ExperiencePageSource = {
  readonly experiences: readonly ExperienceSourceRecord[];
  readonly education: readonly EducationSourceRecord[];
  readonly skillCategories: readonly SkillCategorySourceRecord[];
};

export type ExperienceEngagement = {
  readonly engagement: string;
  readonly location: string | null;
  readonly organization: string;
  readonly period: string | null;
  readonly practiceAreas: readonly string[];
  readonly role: string;
  readonly summary: string;
};

export type ExperienceEducation = {
  readonly degree: string;
  readonly institution: string;
  readonly period: string | null;
};

export type ExperienceExpertiseGroup = {
  readonly category: string;
  readonly skills: readonly string[];
};

export type PracticeMapStage = {
  readonly description: string;
  readonly label: string;
};

export type ExperiencePageContent = {
  readonly engagements: readonly ExperienceEngagement[];
  readonly education: readonly ExperienceEducation[];
  readonly expertise: readonly ExperienceExpertiseGroup[];
  readonly practiceMap: readonly PracticeMapStage[];
};
