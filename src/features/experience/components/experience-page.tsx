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
          <Surface as="aside" padding="lg" variant="accent">
            <div className="flex items-center gap-3 text-primary">
              <Briefcase aria-hidden="true" className="size-5" />
              <p className="eyebrow">Career overview</p>
            </div>
            <dl className="mt-6 divide-y divide-line border-y border-line">
              <div className="grid grid-cols-[5rem_1fr] gap-4 py-4">
                <dt className="font-serif text-3xl font-medium text-primary">
                  {String(engagements.length).padStart(2, "0")}
                </dt>
                <dd className="self-center text-sm leading-6 text-muted">
                  Published professional records
                </dd>
              </div>
              <div className="grid grid-cols-[5rem_1fr] gap-4 py-4">
                <dt className="font-serif text-3xl font-medium text-secondary">
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

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Engagement timeline"
              title="Experience across production-oriented data and AI work."
              description="A public overview of the professional work and practice areas currently represented in the portfolio."
            />
          </Reveal>

          {engagements.length > 0 ? (
            <ol className="mt-12 border-t border-line" aria-label="Professional experience records">
              {engagements.map((engagement, index) => (
                <li
                  key={`${engagement.organization}-${engagement.role}-${index}`}
                  className="grid gap-6 border-b border-line py-9 md:grid-cols-[10rem_minmax(0,1fr)] md:gap-10 md:py-12"
                >
                  <Reveal>
                    <div>
                      {engagement.period ? (
                        <p className="font-mono text-xs font-semibold tracking-[0.1em] text-primary uppercase">
                          {engagement.period}
                        </p>
                      ) : null}
                      <p className={engagement.period ? "mt-3 font-mono text-xs text-subtle" : "font-mono text-xs text-subtle"}>
                        Engagement {String(index + 1).padStart(2, "0")}
                      </p>
                    </div>
                  </Reveal>

                  <Reveal delay={0.05}>
                    <article>
                      <Badge variant="accent">{engagement.engagement}</Badge>
                      <h3 className="mt-5 font-serif text-3xl leading-tight font-medium tracking-[-0.035em] sm:text-4xl">
                        {engagement.role}
                      </h3>
                      <p className="mt-3 text-base font-semibold text-ink">
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
                    </article>
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
        <section className="border-b border-line py-18 sm:py-24 lg:py-28">
          <Container>
            <Reveal>
              <PracticeMap stages={practiceMap} />
            </Reveal>
          </Container>
        </section>
      ) : null}

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Education"
              title="Academic foundations for technical practice."
              description="Published education records provide the academic context represented throughout the portfolio."
            />
          </Reveal>

          {education.length > 0 ? (
            <div className="mt-12 grid gap-4 lg:grid-cols-2">
              {education.map((item, index) => (
                <Reveal key={`${item.degree}-${item.institution}-${index}`} delay={index * 0.06}>
                  <Surface as="article" className="h-full" padding="lg" variant="raised">
                    <div className="flex items-center justify-between gap-4">
                      <span className="grid size-11 place-items-center rounded-full border border-line bg-surface text-primary">
                        <GraduationCap aria-hidden="true" className="size-5" />
                      </span>
                      {item.period ? (
                        <span className="font-mono text-xs font-semibold tracking-[0.08em] text-secondary uppercase">
                          {item.period}
                        </span>
                      ) : null}
                    </div>
                    <h3 className="mt-8 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
                      {item.degree}
                    </h3>
                    <p className="mt-4 text-sm font-semibold leading-6 text-muted">
                      {item.institution}
                    </p>
                  </Surface>
                </Reveal>
              ))}
            </div>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No education records are currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Technical expertise"
              title="A categorized toolkit for data, models, and delivery."
              description="Technologies and methods are grouped by capability area for a clear view of the technical foundation behind the work."
            />
          </Reveal>

          {expertise.length > 0 ? (
            <div className="mt-12 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {expertise.map((group, index) => (
                <Reveal key={`${group.category}-${index}`} className="h-full" delay={(index % 3) * 0.04}>
                  <Surface as="article" className="h-full" padding="lg" variant="subtle">
                    <div className="flex items-center justify-between gap-4 border-b border-line pb-5">
                      <p className="eyebrow text-primary">Capability</p>
                      <span className="font-mono text-xs text-subtle">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                    </div>
                    <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em]">
                      {group.category}
                    </h3>
                    <ul className="mt-6 flex flex-wrap gap-2" aria-label={`${group.category} skills`}>
                      {group.skills.map((skill) => (
                        <li key={skill}>
                          <Badge variant="outline">{skill}</Badge>
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
