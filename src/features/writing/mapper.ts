import type {
  PublicResearchStage,
  WritingOverview,
  WritingOverviewSource,
} from "./types";

const researchStageLabels = {
  RESEARCH_INTEREST: "Research interest",
  EXPLORING: "Exploring",
  EARLY_STAGE: "Early-stage",
  IN_PROGRESS: "In progress",
  WORKING_PAPER: "Working paper",
  SUBMITTED: "Submitted",
  ACCEPTED: "Accepted",
  PUBLISHED: "Published",
  ARCHIVED: "Archived",
} as const satisfies Record<PublicResearchStage, string>;

export function mapWritingOverview(
  source: WritingOverviewSource,
): WritingOverview {
  const direction = source.sourceDirection;

  return {
    sourceDirection: direction
      ? {
          title: direction.title,
          stage: researchStageLabels[direction.stage],
          format: direction.format,
          summary: direction.summary,
          questions: direction.questions.map(({ question }) => question),
        }
      : null,
    // Article publication and detail routes are intentionally deferred to Phase 8.
    entries: [],
  };
}
