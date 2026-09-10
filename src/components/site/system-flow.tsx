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
      className={`group/atlas relative isolate overflow-hidden rounded-3xl border border-line/80 bg-surface/90 shadow-[0_30px_90px_-55px_rgb(var(--ds-shadow-color)/0.5)] ${className}`}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-32 right-0 size-80 rounded-full bg-primary/[0.08] blur-3xl"
      />
      <figcaption className="relative grid gap-6 border-b border-line/80 px-5 py-8 sm:px-8 lg:grid-cols-[minmax(0,0.82fr)_minmax(18rem,1fr)] lg:items-end lg:px-10 lg:py-10">
        <div>
          <p className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-primary">
            {eyebrow}
          </p>
          <TitleHeading className="mt-4 max-w-2xl text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.04em] text-ink sm:text-4xl">
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
          className="absolute top-[5.15rem] right-[9%] left-[9%] hidden h-px bg-gradient-to-r from-primary/15 via-primary/70 to-secondary/15 xl:block"
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
                className="group relative z-10 flex min-h-64 flex-col overflow-hidden rounded-2xl border border-line/80 bg-canvas/95 p-5 transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1.5 hover:border-primary/60 hover:bg-surface hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
              >
                <span
                  aria-hidden="true"
                  className="absolute inset-x-0 top-0 h-0.5 origin-left scale-x-0 bg-gradient-to-r from-primary to-secondary transition-transform duration-300 group-hover:scale-x-100 motion-reduce:transition-none"
                />
                <div className="flex items-start justify-between gap-4">
                  <span className="font-mono text-[0.68rem] font-medium uppercase tracking-[0.14em] text-subtle">
                    Stage {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="grid size-11 place-items-center rounded-full border border-line bg-surface text-primary shadow-sm transition-[border-color,background-color,transform] duration-300 group-hover:rotate-3 group-hover:scale-105 group-hover:border-primary group-hover:bg-highlight motion-reduce:transform-none motion-reduce:transition-none">
                    <Icon
                      aria-hidden="true"
                      className="size-[1.125rem]"
                      strokeWidth={1.7}
                    />
                  </span>
                </div>

                <div className="mt-11">
                  <StepHeading className="text-base font-semibold tracking-[-0.015em] text-ink">
                    {step.label}
                  </StepHeading>
                  <p className="mt-3 text-sm leading-6 text-muted">
                    {step.description}
                  </p>
                </div>

                <div className="mt-auto flex items-center gap-2 pt-7" aria-hidden="true">
                  <span className="size-2 rounded-full bg-primary shadow-[0_0_0_5px_color-mix(in_srgb,var(--ds-primary)_12%,transparent)]" />
                  <span className="h-px flex-1 bg-line transition-colors duration-300 group-hover:bg-primary motion-reduce:transition-none" />
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
