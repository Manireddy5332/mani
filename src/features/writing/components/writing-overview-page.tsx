import { ArrowUpRight, BookOpen, CircleHelp, FileText } from "lucide-react";

import { PageIntro } from "@/components/site/page-intro";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";

import type { WritingOverview } from "../types";

type WritingOverviewPageProps = {
  overview: WritingOverview;
};

export function WritingOverviewPage({ overview }: WritingOverviewPageProps) {
  const { sourceDirection } = overview;
  const inquiryDescription = sourceDirection
    ? sourceDirection.format
      ? `This ${sourceDirection.stage.toLowerCase()} direction is currently framed as “${sourceDirection.format}” and provides a foundation for future research notes and technical writing.`
      : `This ${sourceDirection.stage.toLowerCase()} direction provides a foundation for future research notes and technical writing.`
    : "No current research direction is publicly listed. Verified work can be added through the portfolio content system when it is ready.";

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <PageIntro
        eyebrow="Writing / Research notes"
        title="Writing rooted in active inquiry."
        description="This space will bring together literature-review notes, AI engineering articles, evaluation observations, learning journals, and technical tutorials as the work develops."
        actions={
          <ButtonLink href="/research" variant="outline">
            Explore the research direction
            <ArrowUpRight aria-hidden="true" className="size-4" />
          </ButtonLink>
        }
        aside={
          <Surface
            as="aside"
            className="group relative isolate overflow-hidden border-primary/20 bg-surface/85 shadow-[0_28px_90px_-58px_rgb(79_93_204/0.65)] backdrop-blur-sm"
            padding="lg"
            variant="raised"
          >
            <div
              aria-hidden="true"
              className="foundation-grid pointer-events-none absolute inset-0 -z-10 opacity-35 [mask-image:linear-gradient(to_bottom_right,black,transparent_72%)]"
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute -top-20 -right-16 -z-10 size-48 rounded-full bg-[radial-gradient(circle,var(--ds-glow-primary),transparent_70%)] transition-transform duration-700 ease-out group-hover:scale-110 motion-reduce:transform-none motion-reduce:transition-none"
            />
            <span
              aria-hidden="true"
              className="absolute top-0 left-0 h-1 w-28 bg-gradient-to-r from-primary to-secondary"
            />
            <div className="flex flex-wrap items-center justify-between gap-4">
              <span className="grid size-12 place-items-center rounded-full border border-primary/25 bg-canvas/80 text-primary shadow-[0_12px_32px_-20px_rgb(79_93_204/0.7)] transition-[border-color,transform] duration-300 group-hover:-translate-y-0.5 group-hover:border-primary/50 motion-reduce:transform-none motion-reduce:transition-none">
                <BookOpen aria-hidden="true" className="size-5" strokeWidth={1.7} />
              </span>
              <Badge variant="outline">Notes in preparation</Badge>
            </div>
            <h2 className="mt-8 font-serif text-3xl leading-tight font-medium tracking-[-0.03em] text-ink">
              Research notes are in preparation.
            </h2>
            <p className="mt-5 text-sm leading-7 text-muted">
              {sourceDirection
                ? "The first entries will grow from the current published research direction, with clear dates and status when they are ready."
                : "Verified entries will appear here with clear dates and status when they are ready."}
            </p>
            <div className="mt-8 flex items-center gap-3 border-t border-line pt-6 text-sm text-muted">
              <span className="grid size-8 shrink-0 place-items-center rounded-full border border-secondary/25 bg-secondary/10 text-secondary">
                <FileText aria-hidden="true" className="size-4" />
              </span>
              <span>
                {sourceDirection
                  ? `Current focus: ${sourceDirection.title}.`
                  : "No current research direction is publicly listed."}
              </span>
            </div>
          </Surface>
        }
      />

      <section className="relative isolate py-20 sm:py-24 lg:py-32">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-64 bg-[radial-gradient(circle_at_18%_0%,var(--ds-glow-primary),transparent_60%)]"
        />
        <Container className="relative">
          <Reveal>
            <SectionHeading
              eyebrow="Current inquiry"
              title={
                sourceDirection
                  ? "The questions currently shaping future notes"
                  : "Research notes will follow verified public work"
              }
              description={inquiryDescription}
            />
          </Reveal>

          {sourceDirection ? (
            <div className="mt-12 grid gap-6 lg:grid-cols-[minmax(18rem,0.62fr)_minmax(0,1fr)] lg:items-start lg:gap-10">
              <Reveal className="lg:sticky lg:top-32" delay={0.04}>
                <Surface
                  className="relative isolate h-full overflow-hidden border-primary/25 bg-primary/[0.045]"
                  padding="lg"
                  variant="accent"
                >
                  <div
                    aria-hidden="true"
                    className="foundation-grid pointer-events-none absolute inset-0 -z-10 opacity-30 [mask-image:linear-gradient(to_bottom,black,transparent_85%)]"
                  />
                  <div className="flex items-center justify-between gap-4">
                    <p className="eyebrow text-primary">Review context</p>
                    <span className="font-mono text-[0.68rem] font-semibold tracking-[0.14em] text-subtle uppercase">
                      Source 01
                    </span>
                  </div>
                  <span
                    aria-hidden="true"
                    className="mt-8 block h-px w-20 bg-gradient-to-r from-primary to-secondary"
                  />
                  <h3 className="mt-7 text-balance font-serif text-2xl leading-snug font-medium tracking-[-0.025em] text-ink sm:text-3xl">
                    {sourceDirection.title}
                  </h3>
                  <p className="mt-5 text-sm leading-7 text-muted sm:text-base">
                    {sourceDirection.summary}
                  </p>
                  <div className="mt-8 flex flex-wrap gap-2 border-t border-line/80 pt-6">
                    <Badge variant="accent">{sourceDirection.stage}</Badge>
                    {sourceDirection.format ? (
                      <Badge variant="outline">{sourceDirection.format}</Badge>
                    ) : null}
                  </div>
                </Surface>
              </Reveal>

              <Reveal delay={0.08}>
                <Surface
                  as="section"
                  aria-labelledby="writing-questions-title"
                  className="overflow-hidden border-line/80 bg-surface/65 shadow-[0_24px_70px_-55px_rgb(20_25_35/0.55)]"
                  padding="none"
                >
                  <div className="flex items-center gap-3 border-b border-line bg-canvas/70 px-6 py-5 sm:px-8">
                    <span className="grid size-9 place-items-center rounded-full border border-primary/20 bg-primary/10 text-primary">
                      <CircleHelp
                        aria-hidden="true"
                        className="size-4"
                        strokeWidth={1.7}
                      />
                    </span>
                    <h3
                      id="writing-questions-title"
                      className="text-sm font-semibold tracking-[-0.01em] text-ink"
                    >
                      Questions under review
                    </h3>
                  </div>
                  {sourceDirection.questions.length > 0 ? (
                    <ol className="divide-y divide-line/80 px-5 sm:px-8">
                      {sourceDirection.questions.map((question, index) => (
                        <li
                          key={`${question}-${index}`}
                          className="group grid gap-4 py-6 transition-colors duration-300 hover:bg-primary/[0.025] motion-reduce:transition-none sm:grid-cols-[3rem_1fr] sm:items-start sm:px-1"
                        >
                          <span className="grid size-9 place-items-center rounded-full border border-line bg-canvas font-mono text-[0.68rem] font-semibold text-primary transition-[border-color,background-color,color] duration-300 group-hover:border-primary/35 group-hover:bg-primary/10 motion-reduce:transition-none">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <p className="pt-1 text-sm leading-7 text-ink sm:text-base">
                            {question}
                          </p>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="px-6 py-7 text-sm leading-7 text-muted sm:px-8">
                      No research questions are currently published for this direction.
                    </p>
                  )}
                </Surface>
              </Reveal>
            </div>
          ) : (
            <Reveal delay={0.04}>
              <Surface className="mt-12" padding="lg" variant="subtle">
                <p className="text-sm leading-7 text-muted">
                  No current research direction is published.
                </p>
              </Surface>
            </Reveal>
          )}
        </Container>
      </section>
    </main>
  );
}
