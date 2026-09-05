import { formatPartialPeriod } from "@/lib/public-content/dates";

import { practiceMap } from "./content";
import type {
  ExperiencePageContent,
  ExperiencePageSource,
  PublicExperienceType,
} from "./types";

const experienceTypeLabels = {
  CLIENT_ENGAGEMENT: "Client engagement",
  EMPLOYMENT: "Employment",
  INTERNSHIP: "Internship",
  CONTRACT: "Contract",
  OTHER: "Professional experience",
} as const satisfies Record<PublicExperienceType, string>;

export function mapExperiencePageContent(
  source: ExperiencePageSource,
): ExperiencePageContent {
  const engagements = source.experiences.map((experience) => ({
    engagement: experienceTypeLabels[experience.type],
    location: experience.location,
    organization: experience.organization,
    period: formatPartialPeriod(experience),
    practiceAreas: experience.practiceAreas,
    role: experience.role,
    summary: experience.summary,
  }));

  return {
    engagements,
    education: source.education.map((education) => ({
      degree: education.degree,
      institution: education.institution,
      period: formatPartialPeriod(education),
    })),
    expertise: source.skillCategories
      .map((category) => ({
        category: category.name,
        skills: category.skills.map((skill) => skill.name),
      }))
      .filter((category) => category.skills.length > 0),
    practiceMap: engagements.length > 0 ? practiceMap : [],
  };
}
