import { portfolioContent } from "@/lib/portfolio";

import type { ResearchDetailRecord, ResearchOverview } from "./types";

const currentResearchSlug = "generative-ai-adoption-in-industry";
const currentResearchHref = `/research/${currentResearchSlug}` as const;

const researchDetails = [
  {
    slug: currentResearchSlug,
    href: currentResearchHref,
    isCurrent: true,
    title: "Generative AI adoption in industry",
    stage: portfolioContent.currentResearch.stage,
    format: portfolioContent.currentResearch.format,
    summary: portfolioContent.currentResearch.summary,
    questions: portfolioContent.currentResearch.questions,
    interests: portfolioContent.researchInterests,
    methodology: "Literature and case-study review",
    methodologyDescription:
      "The review examines how companies use generative AI, which models are adopted, and what benefits and constraints are reported.",
    questionContext:
      "The work is organized around five questions about industry use, model adoption, reported value, operational constraints, and gaps in the literature.",
    scopeTitle: "Review rather than model development",
    scopeBoundary:
      "This review examines existing literature and case studies. It is not a project to build or train a new model.",
    evidenceTitle: "Review in its early stage",
    evidenceStatus:
      "The review is in its early stage. No findings or conclusions are presented.",
    publicationStatus:
      "No formal publication record is associated with this research direction yet.",
    atlasTitle: "A clear view of what is defined—and what remains open.",
    atlasDescription:
      "The map separates the review's current questions and method from links, evidence, and conclusions that have not yet been established.",
    atlasSteps: [
      {
        stage: "questions",
        label: "Research questions",
        description: "Five questions currently define the review.",
      },
      {
        stage: "methods",
        label: "Review method",
        description:
          "Literature and case-study review; no new model is being built or trained.",
      },
      {
        stage: "technologies",
        label: "Models & approaches",
        description:
          "LLM comparison is in scope; RAG is a connected research interest.",
      },
      {
        stage: "projects",
        label: "Related project",
        description: "No project relationship is currently established.",
      },
      {
        stage: "evidence",
        label: "Evidence status",
        description:
          "The review is in its early stage. No findings or conclusions are presented.",
      },
    ],
  },
] as const satisfies readonly ResearchDetailRecord[];

const researchOverview = {
  interests: portfolioContent.researchInterests,
  currentDirection: {
    ...portfolioContent.currentResearch,
    href: currentResearchHref,
    isCurrent: true,
    slug: currentResearchSlug,
    title: "Generative AI adoption in industry",
  },
  publications: [],
} as const satisfies ResearchOverview;

/** Verified CV-backed fallback used only when the public database read throws. */
export function getResearchOverview(): ResearchOverview {
  return researchOverview;
}

export function getResearchDetailBySlug(
  slug: string,
): ResearchDetailRecord | undefined {
  return researchDetails.find((record) => record.slug === slug);
}
