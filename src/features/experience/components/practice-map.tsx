import type { PracticeMapStage } from "@/features/experience/types";

export type PracticeMapProps = {
  readonly stages: readonly PracticeMapStage[];
};

/**
 * A generic synthesis of CV-backed practice areas. This is intentionally not
 * presented as the architecture of a particular client system.
 */
export function PracticeMap({ stages }: PracticeMapProps) {
  return (
    <figure aria-labelledby="practice-map-title" className="overflow-hidden rounded-2xl border border-line bg-ink/[0.025]">
      <figcaption className="grid gap-5 border-b border-line p-6 sm:p-8 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.5fr)] lg:items-end lg:p-10">
        <div>
          <p className="eyebrow text-primary">AI systems practice map</p>
          <h2
            id="practice-map-title"
            className="mt-4 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.035em] sm:text-4xl"
          >
            Recurring work across an applied AI lifecycle.
          </h2>
        </div>
        <p className="text-sm leading-7 text-muted">
          A synthesis of practice areas documented across the CV—not a diagram
          of a specific client platform, deployed architecture, or result.
        </p>
      </figcaption>

      <ol className="grid lg:grid-cols-5" aria-label="Applied AI lifecycle areas">
        {stages.map((stage, index) => (
          <li
            key={stage.label}
            className="relative border-b border-line p-6 last:border-b-0 sm:p-8 lg:border-r lg:border-b-0 lg:last:border-r-0"
          >
            <div className="flex items-center justify-between gap-4">
              <span className="font-mono text-xs font-semibold tracking-[0.12em] text-primary">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span aria-hidden="true" className="h-px flex-1 bg-line" />
              <span aria-hidden="true" className="size-2 rounded-full bg-secondary" />
            </div>
            <h3 className="mt-8 font-serif text-2xl font-medium tracking-[-0.03em]">
              {stage.label}
            </h3>
            <p className="mt-3 text-sm leading-7 text-muted">{stage.description}</p>
          </li>
        ))}
      </ol>
    </figure>
  );
}
