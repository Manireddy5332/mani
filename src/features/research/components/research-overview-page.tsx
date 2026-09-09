import {
  ArrowUpRight,
  CircleHelp,
  FileCheck2,
  FileText,
  FlaskConical,
} from "lucide-react";

import { PageIntro } from "@/components/site/page-intro";
import { EvidenceAtlas } from "@/components/site/system-flow";
import {
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";

import type { ResearchOverview } from "../types";

type ResearchOverviewPageProps = {
  overview: ResearchOverview;
};

function ResearchInterestList({
  interests,
}: {
  interests: ResearchOverview["interests"];
}) {
  return (
    <ol className="grid gap-4 sm:grid-cols-2">
      {interests.map((interest, index) => (
        <li
          key={interest}
          className="group relative flex min-h-48 overflow-hidden rounded-2xl border border-line/80 bg-canvas p-6 shadow-[0_18px_55px_-44px_rgb(20_25_35/0.35)] transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-surface hover:shadow-lift focus-within:border-primary/50 motion-reduce:transform-none motion-reduce:transition-none sm:p-7"
        >
          <span
            aria-hidden="true"
            className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-primary/70 to-secondary opacity-60 transition-opacity duration-300 group-hover:opacity-100 motion-reduce:transition-none"
          />
          <span
            aria-hidden="true"
            className="absolute -right-3 -bottom-8 font-serif text-[7rem] leading-none font-medium tracking-[-0.08em] text-primary/[0.045] transition-transform duration-500 group-hover:-translate-x-2 group-hover:-translate-y-2 group-hover:text-primary/[0.08] motion-reduce:transform-none motion-reduce:transition-none"
          >
            {String(index + 1).padStart(2, "0")}
          </span>
          <div className="relative z-10 flex h-full flex-col justify-between">
            <span className="font-mono text-[0.68rem] font-medium tracking-[0.16em] text-subtle uppercase">
              Interest {String(index + 1).padStart(2, "0")}
            </span>
            <h3 className="mt-10 max-w-sm font-serif text-2xl leading-tight font-medium tracking-[-0.025em] text-ink transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none sm:text-[1.7rem]">
              {interest}
            </h3>
          </div>
        </li>
      ))}
    </ol>
  );
}

function PublicationsEmptyState() {
  return (
    <Surface
      as="section"
      aria-labelledby="publications-title"
      className="grid gap-8 overflow-hidden lg:grid-cols-[auto_minmax(0,1fr)_auto] lg:items-center"
      padding="lg"
      variant="subtle"
    >
      <span
        aria-hidden="true"
        className="grid size-14 place-items-center rounded-full border border-line bg-surface text-primary"
      >
        <FileText className="size-6" strokeWidth={1.6} />
      </span>
      <div>
        <p className="eyebrow text-primary">Publications</p>
        <h2
          id="publications-title"
          className="mt-3 font-serif text-3xl leading-tight font-medium tracking-[-0.03em] text-ink"
        >
          No public publication records are listed yet.
        </h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-muted sm:text-base">
          This area will grow only when verified scholarly work reaches a
          defined public stage.
        </p>
      </div>
      <ButtonLink href="/writing" variant="outline">
        Writing direction
        <ArrowUpRight aria-hidden="true" className="size-4" />
      </ButtonLink>
    </Surface>
  );
}

function PublicationList({
  publications,
}: {
  publications: ResearchOverview["publications"];
}) {
  return (
    <section aria-label="Publications">
      <SectionHeading
        eyebrow="Publications"
        title="Public research outputs"
        description="Only publication records marked for public visibility are listed here."
      />
      <ol className="mt-10 grid gap-4 md:grid-cols-2">
        {publications.map((publication, index) => (
          <li key={publication.slug}>
            <Surface
              as="article"
              className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
              padding="lg"
              variant="subtle"
            >
              <span
                aria-hidden="true"
                className="absolute inset-x-0 top-0 h-1 origin-left scale-x-50 bg-gradient-to-r from-primary to-secondary transition-transform duration-500 group-hover:scale-x-100 motion-reduce:transition-none"
              />
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="eyebrow text-primary">{publication.status}</p>
                <div className="flex items-center gap-3">
                  {publication.year ? (
                    <span className="font-mono text-xs text-subtle">
                      {publication.year}
                    </span>
                  ) : null}
                  <span className="font-mono text-[0.65rem] text-subtle/70">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                </div>
              </div>
              <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink">
                {publication.title}
              </h3>
              {publication.authors.length > 0 ? (
                <p className="mt-4 text-sm leading-6 text-muted">
                  {publication.authors.join(", ")}
                </p>
              ) : null}
              {publication.venue ? (
                <p className="mt-3 text-sm font-semibold leading-6 text-ink/75">
                  {publication.venue}
                </p>
              ) : null}
              {publication.href ? (
                <a
                  className="mt-7 inline-flex items-center gap-2 rounded-sm text-sm font-semibold text-primary underline decoration-primary/30 underline-offset-4 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
                  href={publication.href}
                  rel="noopener noreferrer"
                  target="_blank"
                >
                  View publication
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </a>
              ) : null}
            </Surface>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function ResearchOverviewPage({ overview }: ResearchOverviewPageProps) {
  const { currentDirection, interests, publications } = overview;

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <PageIntro
        eyebrow="Research"
        title="Questions grounded in real-world AI practice."
        description="My research direction examines how generative AI is adopted, evaluated, and sustained beyond controlled demonstrations, with particular attention to retrieval, comparison, and reliability."
        aside={
          <Surface
            as="aside"
            className="relative overflow-hidden shadow-lift"
            padding="lg"
            variant="raised"
          >
            <span
              aria-hidden="true"
              className="absolute -top-20 -right-20 size-48 rounded-full bg-primary/10 blur-3xl"
            />
            <span
              aria-hidden="true"
              className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary"
            />
            <p className="eyebrow text-primary">
              {currentDirection
                ? currentDirection.isCurrent
                  ? "Current direction"
                  : "Featured direction"
                : "Research status"}
            </p>
            {currentDirection ? (
              <>
                <dl className="mt-7 divide-y divide-line border-y border-line">
                  <div className="grid gap-2 py-5 sm:grid-cols-[5rem_1fr]">
                    <dt className="font-mono text-[0.68rem] font-medium tracking-[0.14em] text-subtle uppercase">
                      Stage
                    </dt>
                    <dd className="text-sm font-semibold text-ink">
                      {currentDirection.stage}
                    </dd>
                  </div>
                  <div className="grid gap-2 py-5 sm:grid-cols-[5rem_1fr]">
                    <dt className="font-mono text-[0.68rem] font-medium tracking-[0.14em] text-subtle uppercase">
                      Format
                    </dt>
                    <dd className="text-sm leading-6 font-semibold text-ink">
                      {currentDirection.format ?? "Not publicly specified"}
                    </dd>
                  </div>
                </dl>
                <p className="mt-6 text-sm leading-7 text-muted">
                  This page presents the documented questions and scope without
                  treating open work as completed findings.
                </p>
              </>
            ) : (
              <p className="mt-6 border-y border-line py-5 text-sm leading-7 text-muted">
                No current research direction is publicly listed yet.
              </p>
            )}
          </Surface>
        }
      />

      <section className="relative border-b border-line bg-surface/35 py-20 sm:py-24 lg:py-32">
        <Container>
          {currentDirection ? (
            <>
              <Reveal>
                <SectionHeading
                  actions={
                    <ButtonLink href={currentDirection.href} variant="outline">
                      Explore current direction
                      <ArrowUpRight aria-hidden="true" className="size-4" />
                    </ButtonLink>
                  }
                  eyebrow="Current inquiry"
                  title={currentDirection.title}
                  description={currentDirection.summary}
                />
              </Reveal>

              <div className="mt-12 grid gap-6 lg:grid-cols-[minmax(0,0.55fr)_minmax(0,1fr)] lg:gap-10">
                <Reveal delay={0.04}>
                  <Surface
                    className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/45 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                    padding="lg"
                    variant="accent"
                  >
                    <span
                      aria-hidden="true"
                      className="absolute -right-16 -bottom-16 size-52 rounded-full border border-primary/10 bg-primary/[0.035] transition-transform duration-500 group-hover:scale-110 motion-reduce:transform-none motion-reduce:transition-none"
                    />
                    <span className="relative grid size-12 place-items-center rounded-xl border border-primary/20 bg-canvas/70 text-primary shadow-sm">
                      <FlaskConical
                        aria-hidden="true"
                        className="size-6"
                        strokeWidth={1.6}
                      />
                    </span>
                    <h3 className="mt-8 font-serif text-3xl leading-tight font-medium tracking-[-0.03em] text-ink">
                      Scope before conclusions.
                    </h3>
                    <p className="mt-5 text-sm leading-7 text-muted sm:text-base">
                      {currentDirection.format
                        ? `The work is currently structured as ${currentDirection.format.toLowerCase()}.`
                        : "The public record does not yet specify a research format."} This
                      page keeps its open questions distinct from completed findings.
                    </p>
                  </Surface>
                </Reveal>

                <Reveal delay={0.08}>
                  <Surface
                    as="section"
                    aria-labelledby="questions-title"
                    className="overflow-hidden shadow-[0_24px_70px_-55px_rgb(20_25_35/0.4)]"
                    padding="none"
                  >
                    <div className="flex items-center gap-3 border-b border-line px-6 py-5 sm:px-8">
                      <CircleHelp
                        aria-hidden="true"
                        className="size-5 text-primary"
                        strokeWidth={1.7}
                      />
                      <h3
                        id="questions-title"
                        className="text-sm font-semibold tracking-[-0.01em] text-ink"
                      >
                        Questions guiding this direction
                      </h3>
                    </div>
                    {currentDirection.questions.length > 0 ? (
                      <ol className="divide-y divide-line px-6 sm:px-8">
                        {currentDirection.questions.map((question, index) => (
                          <li
                            key={`${index}-${question}`}
                            className="group grid gap-4 py-6 transition-colors duration-200 hover:bg-primary/[0.035] motion-reduce:transition-none sm:grid-cols-[2.5rem_1fr] sm:items-start"
                          >
                            <span className="font-mono text-xs font-semibold text-primary transition-transform duration-200 group-hover:translate-x-1 motion-reduce:transform-none motion-reduce:transition-none">
                              {String(index + 1).padStart(2, "0")}
                            </span>
                            <p className="text-sm leading-7 text-ink sm:text-base">
                              {question}
                            </p>
                          </li>
                        ))}
                      </ol>
                    ) : (
                      <p className="px-6 py-8 text-sm leading-7 text-muted sm:px-8">
                        No public research questions are listed yet.
                      </p>
                    )}
                  </Surface>
                </Reveal>
              </div>
            </>
          ) : (
            <Reveal>
              <Surface padding="lg" variant="subtle">
                <p className="eyebrow text-primary">Current inquiry</p>
                <h2 className="mt-4 font-serif text-3xl font-medium tracking-[-0.03em] text-ink">
                  No public research direction is listed yet.
                </h2>
                <p className="mt-4 text-sm leading-7 text-muted">
                  A verified research record will appear here after it is published.
                </p>
              </Surface>
            </Reveal>
          )}
        </Container>
      </section>

      <section className="border-b border-line py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Research interests"
              title="A focused agenda for deeper study"
              description="These interests connect current professional practice with areas for deeper academic inquiry."
            />
          </Reveal>
          <Reveal delay={0.06} className="mt-12">
            {interests.length > 0 ? (
              <ResearchInterestList interests={interests} />
            ) : (
              <Surface padding="lg" variant="subtle">
                <p className="text-sm leading-7 text-muted">
                  No public research interests are listed yet.
                </p>
              </Surface>
            )}
          </Reveal>
        </Container>
      </section>

      <section className="foundation-grid relative border-b border-line bg-surface/25 py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <EvidenceAtlas
              eyebrow="Evidence Atlas / Research structure"
              title="A transparent path from question to supported evidence."
              description="This framework will connect research questions, methods, technologies, applied projects, and findings as the portfolio's research record develops."
            />
          </Reveal>
        </Container>
      </section>

      <section className="py-20 sm:py-24 lg:py-32">
        <Container>
          {publications.length === 0 ? (
            <PublicationsEmptyState />
          ) : (
            <PublicationList publications={publications} />
          )}
          <div className="mt-8 flex items-center gap-3 text-sm leading-6 text-muted">
            <FileCheck2
              aria-hidden="true"
              className="size-4 shrink-0 text-secondary"
            />
            <p>
              Any future publication will include its status, venue, authorship,
              and supporting links.
            </p>
          </div>
        </Container>
      </section>
    </main>
  );
}
