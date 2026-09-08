import { portfolioContent } from "@/lib/portfolio";

import type {
  ExperienceEngagement,
  ExperiencePageContent,
  PracticeMapStage,
} from "./types";

const practiceAreasByOrganization = {
  USAA: [
    "Generative AI and RAG",
    "Agent workflows",
    "Forecasting",
    "MLOps and monitoring",
    "Data engineering",
  ],
  "Capita Pvt Ltd": [
    "NLP and document processing",
    "Forecasting and classification",
    "ML APIs",
    "Model interpretability",
    "Business intelligence",
  ],
} as const satisfies Record<
  (typeof portfolioContent.experience)[number]["organization"],
  readonly string[]
>;

export const practiceMap = [
  {
    label: "Frame",
    description:
      "Curate data and define statistical, model-performance, and review checks for the work.",
  },
  {
    label: "Build",
    description:
      "Develop machine-learning, NLP, forecasting, retrieval, or generative-AI capabilities.",
  },
  {
    label: "Connect",
    description:
      "Join models to data pipelines, APIs, retrieval layers, and application workflows.",
  },
  {
    label: "Operate",
    description:
      "Support containerized deployment, orchestration, monitoring, and retraining workflows.",
  },
  {
    label: "Evaluate",
    description:
      "Use statistical checks, interpretability, drift monitoring, and governance-oriented review.",
  },
] as const satisfies readonly PracticeMapStage[];

function displayPeriod(period: string) {
  return period.replace(" - ", " \u2014 ");
}

/**
 * Verified CV-backed fallback for database-read failures. It does not expose
 * client metrics or internal architecture.
 */
export function getStaticExperiencePageContent(): ExperiencePageContent {
  return {
    engagements: portfolioContent.experience.map(
      (engagement): ExperienceEngagement => ({
        ...engagement,
        period: displayPeriod(engagement.period),
        practiceAreas: practiceAreasByOrganization[engagement.organization],
      }),
    ),
    education: portfolioContent.education.map((item) => ({
      ...item,
      period: displayPeriod(item.period),
    })),
    expertise: portfolioContent.technicalExpertise,
    practiceMap,
  };
}
