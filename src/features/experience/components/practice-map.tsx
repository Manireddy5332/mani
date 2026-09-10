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
    <figure
      aria-labelledby="practice-map-title"
      className="relative isolate overflow-hidden rounded-[1.75rem] border border-line bg-canvas shadow-[0_32px_100px_-64px_rgb(20_25_35/0.5)]"
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 right-0 -z-10 size-72 rounded-full bg-primary/[0.09] blur-3xl"
      />
      <div
        aria-hidden="true"
        className="foundation-grid pointer-events-none absolute inset-x-0 top-0 -z-10 h-80 opacity-45 [mask-image:linear-gradient(to_bottom,black,transparent)]"
      />

      <figcaption className="grid gap-8 border-b border-line p-7 sm:p-10 lg:grid-cols-[minmax(0,1fr)_minmax(20rem,0.48fr)] lg:items-end lg:p-12">
        <div>
          <div className="flex items-center gap-3">
            <span aria-hidden="true" className="size-2 rounded-full bg-secondary shadow-[0_0_0_5px_var(--color-secondary-soft)]" />
            <p className="eyebrow text-primary">AI systems practice map</p>
          </div>
          <h2
            id="practice-map-title"
            className="mt-5 max-w-3xl text-balance font-serif text-4xl leading-[1.02] font-medium tracking-[-0.045em] sm:text-5xl"
          >
            Recurring work across an applied AI lifecycle.
          </h2>
        </div>
        <p className="border-l-2 border-primary/30 pl-5 text-sm leading-7 text-muted sm:text-base">
          A synthesis of practice areas documented across the CV—not a diagram
          of a specific client platform, deployed architecture, or result.
        </p>
      </figcaption>

      <div className="relative p-4 sm:p-6 lg:p-8">
        <span
          aria-hidden="true"
          className="absolute top-[3.35rem] right-[10%] left-[10%] hidden h-px bg-gradient-to-r from-transparent via-primary/35 to-transparent lg:block"
        />
        <ol
          className="relative grid gap-3 lg:grid-cols-5"
          aria-label="Applied AI lifecycle areas"
        >
          {stages.map((stage, index) => (
            <li
              key={stage.label}
              className="group relative grid grid-cols-[2.75rem_minmax(0,1fr)] gap-4 rounded-2xl border border-line/70 bg-canvas/90 p-5 transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/45 hover:bg-surface hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none sm:p-6 lg:block lg:min-h-72"
            >
              <div className="relative z-10 grid size-11 place-items-center rounded-full border border-primary/30 bg-canvas font-mono text-[0.68rem] font-semibold tracking-[0.12em] text-primary transition-[border-color,background-color,color,transform] duration-300 group-hover:scale-105 group-hover:border-primary group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transform-none motion-reduce:transition-none">
                {String(index + 1).padStart(2, "0")}
              </div>
              <div>
                <p className="font-mono text-[0.65rem] font-semibold tracking-[0.14em] text-subtle uppercase lg:mt-9">
                  Practice area
                </p>
                <h3 className="mt-2 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink transition-colors duration-300 group-hover:text-primary motion-reduce:transition-none">
                  {stage.label}
                </h3>
                <p className="mt-3 text-sm leading-7 text-muted">
                  {stage.description}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </figure>
  );
}
