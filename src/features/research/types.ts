export type PublicationSummary = {
  readonly authors: readonly string[];
  readonly href: string | null;
  readonly slug: string;
  readonly status: string;
  readonly title: string;
  readonly venue: string | null;
  readonly year: number | null;
};

export type ResearchDirection = {
  readonly format: string | null;
  readonly href: `/research/${string}`;
  readonly isCurrent: boolean;
  readonly questions: readonly string[];
  readonly slug: string;
  readonly stage: string;
  readonly summary: string;
  readonly title: string;
};

export type ResearchOverview = {
  readonly interests: readonly string[];
  readonly currentDirection: ResearchDirection | null;
  readonly publications: readonly PublicationSummary[];
};

export type ResearchDetailRecord = ResearchDirection & {
  readonly atlasDescription: string;
  readonly atlasSteps: readonly ResearchEvidenceStep[];
  readonly atlasTitle: string;
  readonly evidenceTitle: string;
  readonly evidenceStatus: string | null;
  readonly interests: readonly string[];
  readonly methodology: string | null;
  readonly methodologyDescription: string | null;
  readonly publicationStatus: string;
  readonly questionContext: string;
  readonly scopeBoundary: string | null;
  readonly scopeTitle: string;
};

export type ResearchEvidenceStep = {
  readonly description: string;
  readonly label: string;
  readonly stage:
    | "questions"
    | "methods"
    | "technologies"
    | "projects"
    | "evidence";
};
