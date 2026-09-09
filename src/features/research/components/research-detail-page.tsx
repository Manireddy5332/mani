import {
  ArrowLeft,
  ArrowRight,
  CircleHelp,
  FileText,
  FlaskConical,
} from "lucide-react";

import { DetailIntro } from "@/components/site/detail-intro";
import { EvidenceAtlas } from "@/components/site/system-flow";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";

import type { ResearchDetailRecord } from "../types";

export type ResearchDetailPageProps = {
  readonly research: ResearchDetailRecord;
};

export function ResearchDetailPage({ research }: ResearchDetailPageProps) {
  const metadata = [
    { label: "Stage", value: research.stage },
    ...(research.format
      ? [{ label: "Format", value: research.format }]
      : []),
    ...(research.methodology
      ? [{ label: "Method", value: research.methodology }]
      : []),
  ];

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <DetailIntro
        actions={
          <>
            <ButtonLink href="/research" size="lg" variant="secondary">
              <ArrowLeft aria-hidden="true" className="size-4" />
              Research overview
            </ButtonLink>
            <ButtonLink href="/contact" size="lg" variant="outline">
              Discuss this direction
              <ArrowRight aria-hidden="true" className="size-4" />
            </ButtonLink>
          </>
        }
        currentLabel={research.title}
        description={research.summary}
        eyebrow="Research direction"
        metadata={metadata}
        parentHref="/research"
        parentLabel="Research"
        status={research.stage}
        title={research.title}
      />

      <section className="relative border-b border-line bg-surface/35 py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Current inquiry"
              title="Questions guiding this direction"
              description={research.questionContext}
            />
          </Reveal>

          {research.questions.length > 0 ? (
            <ol
              className="mt-12 grid gap-4 lg:grid-cols-2"
              aria-label="Research questions"
            >
              {research.questions.map((question, index) => (
                <li
                  key={`${index}-${question}`}
                  className="group relative min-h-64 overflow-hidden rounded-2xl border border-line/80 bg-canvas p-6 shadow-[0_20px_60px_-48px_rgb(20_25_35/0.4)] transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none sm:p-8"
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-1 origin-left scale-x-50 bg-gradient-to-r from-primary to-secondary transition-transform duration-500 group-hover:scale-x-100 motion-reduce:transition-none"
                  />
                  <span
                    aria-hidden="true"
                    className="absolute -right-4 -bottom-10 font-serif text-[8rem] leading-none font-medium tracking-[-0.08em] text-primary/[0.04] transition-transform duration-500 group-hover:-translate-x-2 group-hover:-translate-y-2 motion-reduce:transform-none motion-reduce:transition-none"
                  >
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div className="relative flex items-center justify-between gap-4 border-b border-line pb-5">
                    <span className="font-mono text-xs font-semibold tracking-[0.12em] text-primary uppercase">
                      Question {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="grid size-9 place-items-center rounded-full border border-secondary/20 bg-secondary/10 text-secondary transition-transform duration-300 group-hover:rotate-6 group-hover:scale-105 motion-reduce:transform-none motion-reduce:transition-none">
                      <CircleHelp aria-hidden="true" className="size-4" />
                    </span>
                  </div>
                  <h3 className="relative mt-7 max-w-xl font-serif text-2xl leading-tight font-medium tracking-[-0.025em] text-ink transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none sm:text-3xl">
                    {question}
                  </h3>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No public research questions are listed for this direction yet.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="border-b border-line py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Scope and method"
              title="Scope, method, and evidence boundaries made explicit"
              description="This page separates the documented method and scope from any evidence or publication status recorded for the work."
            />
          </Reveal>

          <div className="mt-12 grid gap-5 lg:grid-cols-3">
            <Reveal className="h-full">
              <Surface
                as="article"
                className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/45 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                padding="lg"
                variant="accent"
              >
                <span className="grid size-12 place-items-center rounded-xl border border-primary/20 bg-canvas/70 text-primary shadow-sm transition-transform duration-300 group-hover:-rotate-3 group-hover:scale-105 motion-reduce:transform-none motion-reduce:transition-none">
                  <FlaskConical aria-hidden="true" className="size-6" />
                </span>
                <p className="mt-8 eyebrow text-primary">Method</p>
                <h3 className="mt-3 font-serif text-3xl leading-tight font-medium tracking-[-0.03em]">
                  {research.methodology ?? "Method not yet documented"}
                </h3>
                <p className="mt-5 text-sm leading-7 text-muted">
                  {research.methodologyDescription ??
                    "No public methodology summary is listed yet."}
                </p>
              </Surface>
            </Reveal>

            <Reveal className="h-full" delay={0.05}>
              <Surface
                as="article"
                className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-secondary/45 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                padding="lg"
                variant="subtle"
              >
                <span className="grid size-12 place-items-center rounded-xl border border-secondary/20 bg-secondary/10 text-secondary shadow-sm transition-transform duration-300 group-hover:rotate-3 group-hover:scale-105 motion-reduce:transform-none motion-reduce:transition-none">
                  <CircleHelp aria-hidden="true" className="size-6" />
                </span>
                <p className="mt-8 eyebrow text-secondary">Study boundary</p>
                <h3 className="mt-3 font-serif text-3xl leading-tight font-medium tracking-[-0.03em]">
                  {research.scopeTitle}
                </h3>
                <p className="mt-5 text-sm leading-7 text-muted">
                  {research.scopeBoundary ??
                    "No public scope boundary is listed yet."}
                </p>
              </Surface>
            </Reveal>

            <Reveal className="h-full" delay={0.1}>
              <Surface
                as="article"
                className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                padding="lg"
                variant="raised"
              >
                <span className="grid size-12 place-items-center rounded-xl border border-primary/20 bg-primary/10 text-primary shadow-sm transition-transform duration-300 group-hover:-rotate-3 group-hover:scale-105 motion-reduce:transform-none motion-reduce:transition-none">
                  <FileText aria-hidden="true" className="size-6" />
                </span>
                <p className="mt-8 eyebrow text-primary">Current status</p>
                <h3 className="mt-3 font-serif text-3xl leading-tight font-medium tracking-[-0.03em]">
                  {research.evidenceTitle}
                </h3>
                <p className="mt-5 text-sm leading-7 text-muted">
                  {research.evidenceStatus ??
                    "No public evidence statement is listed yet."}{" "}
                  {research.publicationStatus}
                </p>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section className="foundation-grid relative border-b border-line bg-surface/25 py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <EvidenceAtlas
              eyebrow="Evidence Atlas / Current research direction"
              title={research.atlasTitle}
              description={research.atlasDescription}
              steps={research.atlasSteps}
            />
          </Reveal>
        </Container>
      </section>

      <section className="py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Wider research agenda"
              title="Connected areas for continued inquiry"
              description="These are the broader research interests documented alongside this research direction."
            />
          </Reveal>

          {research.interests.length > 0 ? (
            <ol className="mt-12 grid gap-4 sm:grid-cols-2">
              {research.interests.map((interest, index) => (
                <li key={`${index}-${interest}`}>
                  <Reveal className="h-full" delay={index * 0.04}>
                    <Surface
                      as="article"
                      className="group relative h-full min-h-48 overflow-hidden transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary opacity-60 transition-opacity duration-300 group-hover:opacity-100 motion-reduce:transition-none"
                      />
                      <Badge variant="outline">
                        Interest {String(index + 1).padStart(2, "0")}
                      </Badge>
                      <h3 className="mt-7 font-serif text-3xl leading-tight font-medium tracking-[-0.03em] transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none">
                        {interest}
                      </h3>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No public research interests are linked to this direction yet.
              </p>
            </Surface>
          )}

          <Reveal className="mt-10">
            <Surface
              as="aside"
              aria-label="Research overview navigation"
              className="relative flex flex-col gap-6 overflow-hidden shadow-[0_24px_70px_-52px_rgb(20_25_35/0.45)] sm:flex-row sm:items-center sm:justify-between"
              padding="lg"
              variant="accent"
            >
              <div>
                <p className="eyebrow text-primary">Research overview</p>
                <p className="mt-3 max-w-2xl text-base leading-7 text-muted">
                  Return to the research page for the complete research
                  interests and current research context.
                </p>
              </div>
              <ButtonLink href="/research" variant="secondary">
                View research overview
                <ArrowRight aria-hidden="true" className="size-4" />
              </ButtonLink>
            </Surface>
          </Reveal>
        </Container>
      </section>
    </main>
  );
}
