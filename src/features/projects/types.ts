export type ProjectContribution = {
  readonly label: string;
  readonly description: string;
};

export type ProjectPageRecord = {
  readonly advisor: string | null;
  readonly contributions: readonly ProjectContribution[];
  readonly implementation: string | null;
  readonly institution: string | null;
  readonly repository: string | null;
  readonly role: string | null;
  readonly href: `/projects/${string}`;
  readonly indexSummaryDescription: string;
  readonly indexSummaryTitle: string;
  readonly seoDescription: string | null;
  readonly seoTitle: string | null;
  readonly shortTitle: string;
  readonly slug: string;
  readonly summary: string;
  readonly technologies: readonly string[];
  readonly title: string;
  readonly type: string;
};

export type ProjectDetailRecord = ProjectPageRecord & {
  readonly caseStudy: {
    readonly contributionsDescription: string;
    readonly contributionsTitle: string;
    readonly learningDescription: string;
    readonly learningTitle: string;
    readonly overviewTitle: string;
    readonly technologyDescription: string;
    readonly technologyTitle: string;
  };
  readonly learningFeatures: readonly string[];
};

export type ProjectsPageContent = {
  readonly projects: readonly ProjectPageRecord[];
};
