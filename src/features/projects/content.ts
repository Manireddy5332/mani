import { portfolioContent } from "@/lib/portfolio";

import type {
  ProjectContribution,
  ProjectDetailRecord,
  ProjectsPageContent,
} from "./types";

export const PROJECT_DETAIL_SLUG = "discover-learn-and-protect" as const;

const selectedContributions = [
  {
    label: "Leadership and coordination",
    description:
      "Coordinated project planning, task allocation, weekly meetings, communication with the project owner, and support across the team lifecycle.",
  },
  {
    label: "Application development",
    description:
      "Contributed to front-end and back-end development, database design, feature implementation, and integration of application components.",
  },
  {
    label: "Learning experience",
    description:
      "Helped shape quizzes, video lectures, reading assignments, progress tracking, and AI-assisted learning for biodiversity education.",
  },
  {
    label: "Integrations and quality",
    description:
      "Integrated YouTube and OpenAI ChatGPT APIs and participated in testing, debugging, feature enhancement, and the final presentation.",
  },
] as const satisfies readonly ProjectContribution[];

const learningFeatures = [
  "Quizzes",
  "Video lectures",
  "Reading assignments",
  "Progress tracking",
  "AI-assisted learning capabilities",
] as const;

const projectDetailRecords = [
  {
    ...portfolioContent.selectedProject,
    slug: PROJECT_DETAIL_SLUG,
    href: `/projects/${PROJECT_DETAIL_SLUG}`,
    shortTitle: "Discover, Learn, and Protect",
    indexSummaryTitle:
      "A capstone connecting learning, technology, and the environment.",
    indexSummaryDescription:
      "The selected project combines biodiversity education, interactive web development, external content, and AI-assisted learning.",
    implementation: "Interactive educational web application",
    repository: portfolioContent.links.projectRepository,
    seoDescription: null,
    seoTitle: null,
    contributions: selectedContributions,
    learningFeatures,
    caseStudy: {
      overviewTitle:
        "An educational experience for biodiversity and environmental learning.",
      learningTitle: "Features supporting informal STEM learning.",
      learningDescription:
        "The application brought several learning formats into one interactive web experience.",
      contributionsTitle: "Responsibilities across the project lifecycle.",
      contributionsDescription:
        "The capstone work covered team coordination, application development, learning-experience features, integrations, and quality work.",
      technologyTitle: "Tools used in the capstone.",
      technologyDescription:
        "The technologies used in this academic capstone.",
    },
  },
] as const satisfies readonly ProjectDetailRecord[];

/** Looks up a public project record without creating fallback content. */
export function getProjectDetailBySlug(
  slug: string,
): ProjectDetailRecord | undefined {
  return projectDetailRecords.find((project) => project.slug === slug);
}

/** Verified CV-backed fallback used only when the public database read throws. */
export function getProjectsPageContent(): ProjectsPageContent {
  return {
    projects: projectDetailRecords,
  };
}
