import type {
  PublicationSummary,
  ResearchDetailRecord,
  ResearchDirection,
  ResearchOverview,
} from "./types";

export type PublicResearchInterestRow = Readonly<{
  id: string;
  name: string;
}>;

export type PublicResearchQuestionRow = Readonly<{
  id: string;
  question: string;
}>;

export type PublicPublicationRow = Readonly<{
  arxivUrl: string | null;
  authors: readonly string[];
  id: string;
  paperUrl: string | null;
  slug: string;
  stage: string;
  title: string;
  venue: string | null;
  year: number | null;
}>;

export type PublicRelatedProjectRow = Readonly<{
  id: string;
  shortTitle: string | null;
  title: string;
}>;

export type PublicResearchDirectionRow = Readonly<{
  format: string | null;
  id: string;
  isCurrent: boolean;
  questions: readonly PublicResearchQuestionRow[];
  slug: string;
  stage: string;
  summary: string;
  title: string;
}>;

export type PublicResearchProjectRow = PublicResearchDirectionRow &
  Readonly<{
    evidenceStatus: string | null;
    interests: readonly PublicResearchInterestRow[];
    methodology: string | null;
    methodologySummary: string | null;
    projects: readonly PublicRelatedProjectRow[];
    publications: readonly PublicPublicationRow[];
    scopeBoundary: string | null;
    technologies: readonly string[];
  }>;

export type PublicResearchOverviewRow = Readonly<{
  interests: readonly PublicResearchInterestRow[];
  currentDirection: PublicResearchDirectionRow | null;
  publications: readonly PublicPublicationRow[];
}> | null;

const researchStageLabels: Readonly<Record<string, string>> = {
  ACCEPTED: "Accepted",
  ARCHIVED: "Archived",
  EARLY_STAGE: "Early-stage",
  EXPLORING: "Exploring",
  IN_PROGRESS: "In progress",
  PUBLISHED: "Published",
  RESEARCH_INTEREST: "Research interest",
  SUBMITTED: "Submitted",
  WORKING_PAPER: "Working paper",
};

const publicationStageLabels: Readonly<Record<string, string>> = {
  ACCEPTED: "Accepted",
  ARCHIVED: "Archived",
  PUBLISHED: "Published",
  SUBMITTED: "Submitted",
  WORKING_PAPER: "Working paper",
};

function labelEnum(value: string, labels: Readonly<Record<string, string>>) {
  return labels[value] ?? value.toLowerCase().replaceAll("_", " ");
}

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

export function mapPublicationSummary(
  publication: PublicPublicationRow,
): PublicationSummary {
  return {
    authors: [...publication.authors],
    href:
      publicHttpUrl(publication.paperUrl) ??
      publicHttpUrl(publication.arxivUrl),
    slug: publication.slug,
    status: labelEnum(publication.stage, publicationStageLabels),
    title: publication.title,
    venue: publication.venue,
    year: publication.year,
  };
}

export function mapResearchDirection(
  research: PublicResearchDirectionRow,
): ResearchDirection {
  return {
    format: research.format,
    href: `/research/${research.slug}`,
    isCurrent: research.isCurrent,
    questions: research.questions.map((item) => item.question),
    slug: research.slug,
    stage: labelEnum(research.stage, researchStageLabels),
    summary: research.summary,
    title: research.title,
  };
}

export function mapResearchOverview(
  row: PublicResearchOverviewRow,
): ResearchOverview {
  if (!row) {
    return { currentDirection: null, interests: [], publications: [] };
  }

  return {
    currentDirection: row.currentDirection
      ? mapResearchDirection(row.currentDirection)
      : null,
    interests: row.interests.map((interest) => interest.name),
    publications: row.publications.map(mapPublicationSummary),
  };
}

function countLabel(count: number, singular: string, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function mapResearchDetail(
  research: PublicResearchProjectRow,
): ResearchDetailRecord {
  const direction = mapResearchDirection(research);
  const questionCount = research.questions.length;
  const projectCount = research.projects.length;
  const publicationCount = research.publications.length;
  const technologyCount = research.technologies.length;

  const publicationStatus =
    publicationCount > 0
      ? `${countLabel(publicationCount, "related public publication")} listed.`
      : "No formal publication record is associated with this research direction yet.";

  return {
    ...direction,
    atlasDescription:
      "This map keeps the documented questions, methods, technologies, related work, and evidence status distinct.",
    atlasSteps: [
      {
        stage: "questions",
        label: "Research questions",
        description:
          questionCount > 0
            ? `${countLabel(questionCount, "question")} currently ${questionCount === 1 ? "defines" : "define"} this direction.`
            : "No public research questions are listed yet.",
      },
      {
        stage: "methods",
        label: "Method",
        description:
          research.methodologySummary ??
          research.methodology ??
          "No public methodology statement is listed yet.",
      },
      {
        stage: "technologies",
        label: "Technologies",
        description:
          technologyCount > 0
            ? `${countLabel(technologyCount, "technology", "technologies")} ${technologyCount === 1 ? "is" : "are"} documented for this direction.`
            : "No technologies are publicly associated with this direction yet.",
      },
      {
        stage: "projects",
        label: "Related projects",
        description:
          projectCount > 0
            ? `${countLabel(projectCount, "related public project")} documented.`
            : "No public project relationship is currently established.",
      },
      {
        stage: "evidence",
        label: "Evidence status",
        description:
          research.evidenceStatus ??
          "No public evidence statement is listed yet.",
      },
    ],
    atlasTitle: "A clear view of what is documented—and what remains open.",
    evidenceTitle: research.evidenceStatus
      ? "Current evidence status"
      : "Evidence not yet documented",
    evidenceStatus: research.evidenceStatus,
    interests: research.interests.map((interest) => interest.name),
    methodology: research.methodology,
    methodologyDescription: research.methodologySummary,
    publicationStatus,
    questionContext:
      questionCount > 0
        ? `${countLabel(questionCount, "public question")} currently ${questionCount === 1 ? "guides" : "guide"} this research direction.`
        : "No public research questions are listed for this direction yet.",
    scopeBoundary: research.scopeBoundary,
    scopeTitle: research.scopeBoundary
      ? "Documented study boundary"
      : "Scope not yet documented",
  };
}
