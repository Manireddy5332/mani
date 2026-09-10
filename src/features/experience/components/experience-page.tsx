import { Briefcase, GraduationCap } from "lucide-react";

import { PageIntro } from "@/components/site/page-intro";
import {
  Badge,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import type { ExperiencePageContent } from "@/features/experience/types";
import { PracticeMap } from "@/features/experience/components/practice-map";

export type ExperiencePageProps = {
  readonly content: ExperiencePageContent;
};

export function ExperiencePage({ content }: ExperiencePageProps) {
  const { education, engagements, expertise, practiceMap } = content;

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <PageIntro
        eyebrow="Professional experience"
        title="Applied AI work across the systems lifecycle."
        description="A public, confidentiality-conscious overview of professional experience, education, and technical expertise."
        aside={
          <Surface
            as="aside"
            className="relative isolate overflow-hidden"
            padding="lg"
            variant="accent"
          >
            <div
              aria-hidden="true"
              className="pointer-events-none absolute -top-12 -right-12 -z-10 size-40 rounded-full bg-primary/10 blur-2xl"
            />
            <div className="flex items-center gap-3 text-primary">
              <span className="grid size-10 place-items-center rounded-full border border-primary/25 bg-canvas/70">
                <Briefcase aria-hidden="true" className="size-4" />
              </span>
              <p className="eyebrow">Career overview</p>
            </div>
            <dl className="mt-7 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-1">
              <div className="grid grid-cols-[4.5rem_1fr] gap-3 bg-canvas/75 p-4">
                <dt className="font-serif text-4xl leading-none font-medium tracking-[-0.04em] text-primary">
                  {String(engagements.length).padStart(2, "0")}
                </dt>
                <dd className="self-center text-sm leading-6 text-muted">
                  Published professional records
                </dd>
              </div>
              <div className="grid grid-cols-[4.5rem_1fr] gap-3 bg-canvas/75 p-4">
                <dt className="font-serif text-4xl leading-none font-medium tracking-[-0.04em] text-secondary">
                  {String(education.length).padStart(2, "0")}
                </dt>
                <dd className="self-center text-sm leading-6 text-muted">
                  Published education records
                </dd>
              </div>
            </dl>
          </Surface>
        }
      />

      <section className="relative border-b border-line py-20 sm:py-28 lg:py-32">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 right-0 hidden w-1/3 bg-gradient-to-l from-primary/[0.035] to-transparent lg:block"
        />
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Engagement timeline"
              title="Experience across production-oriented data and AI work."
              description="A public overview of the professional work and practice areas currently represented in the portfolio."
            />
          </Reveal>

          {engagements.length > 0 ? (
            <ol
              className="relative mt-14 space-y-8 before:absolute before:top-6 before:bottom-6 before:left-[1.35rem] before:w-px before:bg-gradient-to-b before:from-primary/60 before:via-line-strong before:to-transparent before:content-[''] md:mt-16 md:space-y-10 md:before:left-[11.35rem]"
              aria-label="Professional experience records"
            >
              {engagements.map((engagement, index) => (
                <li
                  key={`${engagement.organization}-${engagement.role}-${index}`}
                  className="relative grid grid-cols-[2.75rem_minmax(0,1fr)] gap-4 md:grid-cols-[10rem_2.75rem_minmax(0,1fr)] md:gap-5"
                >
                  <Reveal className="hidden pt-6 text-right md:block">
                    {engagement.period ? (
                      <p className="font-mono text-xs font-semibold tracking-[0.1em] text-primary uppercase">
                        {engagement.period}
                      </p>
                    ) : null}
                    <p
                      className={
                        engagement.period
                          ? "mt-3 font-mono text-[0.68rem] tracking-[0.08em] text-subtle uppercase"
                          : "font-mono text-[0.68rem] tracking-[0.08em] text-subtle uppercase"
                      }
                    >
                      Engagement {String(index + 1).padStart(2, "0")}
                    </p>
                  </Reveal>

                  <div className="relative z-10 flex justify-center pt-5">
                    <span className="grid size-11 place-items-center rounded-full border border-primary/35 bg-canvas shadow-[0_0_0_7px_var(--color-canvas)]">
                      <span
                        aria-hidden="true"
                        className="size-2.5 rounded-full bg-primary shadow-[0_0_0_4px_rgb(71_84_194/0.12)]"
                      />
                      <span className="sr-only">Timeline point {index + 1}</span>
                    </span>
                  </div>

                  <Reveal delay={Math.min(index * 0.04, 0.16)}>
                    <Surface
                      as="article"
                      className="group relative overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant={index === 0 ? "raised" : "outlined"}
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-primary/40 to-secondary/40"
                      />
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <Badge variant="accent">{engagement.engagement}</Badge>
                        {engagement.period ? (
                          <p className="font-mono text-[0.68rem] font-semibold tracking-[0.1em] text-primary uppercase md:hidden">
                            {engagement.period}
                          </p>
                        ) : null}
                      </div>
                      <h3 className="mt-6 text-balance font-serif text-3xl leading-[1.05] font-medium tracking-[-0.04em] text-ink transition-colors duration-300 group-hover:text-primary motion-reduce:transition-none sm:text-4xl">
                        {engagement.role}
                      </h3>
                      <p className="mt-3 text-base font-semibold text-ink/90">
                        {engagement.organization}
                        {engagement.location ? (
                          <span className="font-normal text-muted">
                            {" "}/ {engagement.location}
                          </span>
                        ) : null}
                      </p>
                      <p className="mt-6 max-w-3xl text-base leading-8 text-muted">
                        {engagement.summary}
                      </p>
                      {engagement.practiceAreas.length > 0 ? (
                        <ul className="mt-7 flex flex-wrap gap-2" aria-label={`${engagement.organization} practice areas`}>
                          {engagement.practiceAreas.map((area) => (
                            <li key={area}>
                              <Badge variant="outline">{area}</Badge>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No professional experience is currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      {practiceMap.length > 0 ? (
        <section className="border-b border-line bg-surface/45 py-20 sm:py-28 lg:py-32">
          <Container>
            <Reveal>
              <PracticeMap stages={practiceMap} />
            </Reveal>
          </Container>
        </section>
      ) : null}

      <section className="border-b border-line py-20 sm:py-28 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Education"
              title="Academic foundations for technical practice."
              description="Published education records provide the academic context represented throughout the portfolio."
            />
          </Reveal>

          {education.length > 0 ? (
            <ol className="mt-14 grid gap-5 lg:grid-cols-2">
              {education.map((item, index) => (
                <li key={`${item.degree}-${item.institution}-${index}`}>
                  <Reveal className="h-full" delay={index * 0.06}>
                    <Surface
                      as="article"
                      className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="raised"
                    >
                      <div
                        aria-hidden="true"
                        className="absolute -top-16 -right-16 size-44 rounded-full bg-secondary/[0.08] blur-2xl"
                      />
                      <div className="relative flex items-center justify-between gap-4">
                        <span className="grid size-12 place-items-center rounded-2xl border border-primary/25 bg-primary/[0.065] text-primary transition-[background-color,color,transform] duration-300 group-hover:scale-105 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transform-none motion-reduce:transition-none">
                        <GraduationCap aria-hidden="true" className="size-5" />
                        </span>
                        <span className="font-mono text-[0.68rem] font-semibold tracking-[0.12em] text-subtle uppercase">
                          Academic record {String(index + 1).padStart(2, "0")}
                        </span>
                      </div>
                      <h3 className="relative mt-9 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.04em]">
                        {item.degree}
                      </h3>
                      <div className="relative mt-6 flex flex-wrap items-end justify-between gap-4 border-t border-line pt-5">
                        <p className="text-sm font-semibold leading-6 text-muted">
                          {item.institution}
                        </p>
                        {item.period ? (
                          <span className="font-mono text-xs font-semibold tracking-[0.08em] text-secondary uppercase">
                            {item.period}
                          </span>
                        ) : null}
                      </div>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No education records are currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="bg-surface/35 py-20 sm:py-28 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Technical expertise"
              title="A categorized toolkit for data, models, and delivery."
              description="Technologies and methods are grouped by capability area for a clear view of the technical foundation behind the work."
            />
          </Reveal>

          {expertise.length > 0 ? (
            <div className="mt-14 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {expertise.map((group, index) => (
                <Reveal key={`${group.category}-${index}`} className="h-full" delay={(index % 3) * 0.04}>
                  <Surface
                    as="article"
                    className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                    padding="lg"
                    variant="subtle"
                  >
                    <div className="flex items-center gap-4 border-b border-line pb-5">
                      <span className="grid size-9 shrink-0 place-items-center rounded-full border border-primary/25 bg-primary/[0.06] font-mono text-[0.65rem] font-semibold text-primary transition-colors duration-300 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <p className="eyebrow text-primary">Capability group</p>
                    </div>
                    <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em]">
                      {group.category}
                    </h3>
                    <ul className="mt-6 flex flex-wrap gap-2" aria-label={`${group.category} skills`}>
                      {group.skills.map((skill) => (
                        <li key={skill}>
                          <Badge
                            className="transition-[border-color,background-color] duration-200 group-hover:border-primary/30 group-hover:bg-primary/[0.035] motion-reduce:transition-none"
                            variant="outline"
                          >
                            {skill}
                          </Badge>
                        </li>
                      ))}
                    </ul>
                  </Surface>
                </Reveal>
              ))}
            </div>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No technical expertise categories are currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>
    </main>
  );
}
