import type { LucideIcon } from "lucide-react";
import {
  Blocks,
  CircleHelp,
  FileCheck2,
  FlaskConical,
  FolderKanban,
} from "lucide-react";

export type EvidenceAtlasStage =
  | "questions"
  | "methods"
  | "technologies"
  | "projects"
  | "evidence";

export type EvidenceAtlasStep = {
  stage: EvidenceAtlasStage;
  label: string;
  description: string;
};

const iconByStage = {
  questions: CircleHelp,
  methods: FlaskConical,
  technologies: Blocks,
  projects: FolderKanban,
  evidence: FileCheck2,
} satisfies Record<EvidenceAtlasStage, LucideIcon>;

export const defaultEvidenceAtlasSteps = [
  {
    stage: "questions",
    label: "Research questions",
    description: "Frame what needs to be understood, compared, or tested.",
  },
  {
    stage: "methods",
    label: "Methods",
    description: "Document study design, evaluation criteria, and limits.",
  },
  {
    stage: "technologies",
    label: "Technologies",
    description:
      "Map the models, retrieval patterns, and tools under examination.",
  },
  {
    stage: "projects",
    label: "Projects",
    description: "Connect each inquiry to the work where it is explored.",
  },
  {
    stage: "evidence",
    label: "Findings & evidence",
    description:
      "Surface supported outcomes while keeping open questions explicit.",
  },
] as const satisfies readonly EvidenceAtlasStep[];

export type EvidenceAtlasProps = {
  className?: string;
  eyebrow?: string;
  title?: string;
  description?: string;
  steps?: readonly EvidenceAtlasStep[];
  headingLevel?: "h2" | "h3";
};

export function EvidenceAtlas({
  className = "",
  eyebrow = "Evidence atlas / Research map",
  title = "Trace questions to evidence—without skipping the middle.",
  description =
    "A structure for connecting research intent, study design, technical choices, applied work, and supported outcomes.",
  steps = defaultEvidenceAtlasSteps,
  headingLevel = "h2",
}: EvidenceAtlasProps) {
  const TitleHeading = headingLevel;
  const StepHeading = headingLevel === "h2" ? "h3" : "h4";

  return (
    <figure
      className={`overflow-hidden rounded-lg border border-line bg-surface shadow-soft ${className}`}
    >
      <figcaption className="grid gap-5 border-b border-line px-5 py-7 sm:px-7 lg:grid-cols-[minmax(0,0.75fr)_minmax(18rem,1fr)] lg:items-end lg:px-9 lg:py-9">
        <div>
          <p className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-primary">
            {eyebrow}
          </p>
          <TitleHeading className="mt-3 max-w-2xl text-balance text-2xl font-semibold tracking-[-0.035em] text-ink sm:text-3xl">
            {title}
          </TitleHeading>
        </div>
        <p className="max-w-xl text-sm leading-6 text-muted sm:text-base sm:leading-7 lg:justify-self-end">
          {description}
        </p>
      </figcaption>

      <div className="foundation-grid relative p-4 sm:p-6 lg:p-8">
        <div
          aria-hidden="true"
          className="absolute left-[10%] right-[10%] top-[4.7rem] hidden h-px bg-line-strong xl:block"
        />
        <ol
          aria-label="Evidence Atlas research map"
          className="relative grid gap-3 md:grid-cols-2 xl:grid-cols-5"
        >
          {steps.map((step, index) => {
            const Icon = iconByStage[step.stage];

            return (
              <li
                key={`${step.stage}-${index}`}
                className="group relative z-10 flex min-h-60 flex-col border border-line bg-canvas/95 p-5 transition-[border-color,transform,box-shadow] duration-200 hover:-translate-y-1 hover:border-primary hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
              >
                <div className="flex items-start justify-between gap-4">
                  <span className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.14em] text-subtle">
                    Stage {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="grid size-10 place-items-center rounded-full border border-line bg-surface text-primary transition-colors duration-200 group-hover:border-primary group-hover:bg-highlight motion-reduce:transition-none">
                    <Icon
                      aria-hidden="true"
                      className="size-[1.125rem]"
                      strokeWidth={1.7}
                    />
                  </span>
                </div>

                <div className="mt-10">
                  <StepHeading className="text-base font-semibold tracking-[-0.015em] text-ink">
                    {step.label}
                  </StepHeading>
                  <p className="mt-3 text-sm leading-6 text-muted">
                    {step.description}
                  </p>
                </div>

                <div className="mt-auto flex items-center gap-2 pt-7" aria-hidden="true">
                  <span className="size-1.5 rounded-full bg-primary" />
                  <span className="h-px flex-1 bg-line transition-colors duration-200 group-hover:bg-primary motion-reduce:transition-none" />
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </figure>
  );
}

/** Backward-compatible name for the Phase 2 composition. */
export const SystemFlow = EvidenceAtlas;
