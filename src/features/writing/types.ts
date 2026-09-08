export type WritingEntrySummary = {
  readonly title: string;
  readonly slug: string;
  readonly excerpt: string;
};

export type PublicResearchStage =
  | "RESEARCH_INTEREST"
  | "EXPLORING"
  | "EARLY_STAGE"
  | "IN_PROGRESS"
  | "WORKING_PAPER"
  | "SUBMITTED"
  | "ACCEPTED"
  | "PUBLISHED"
  | "ARCHIVED";

export type WritingResearchSource = {
  readonly title: string;
  readonly stage: PublicResearchStage;
  readonly format: string | null;
  readonly summary: string;
  readonly questions: readonly {
    readonly question: string;
  }[];
};

export type WritingOverviewSource = {
  readonly sourceDirection: WritingResearchSource | null;
};

export type WritingResearchDirection = {
  readonly title: string;
  readonly stage: string;
  readonly format: string | null;
  readonly summary: string;
  readonly questions: readonly string[];
};

export type WritingOverview = {
  readonly sourceDirection: WritingResearchDirection | null;
  readonly entries: readonly WritingEntrySummary[];
};
