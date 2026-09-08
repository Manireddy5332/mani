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
          <Surface as="aside" className="relative overflow-hidden" padding="lg" variant="raised">
            <span
              aria-hidden="true"
              className="absolute top-0 left-0 h-1 w-28 bg-primary"
            />
            <div className="flex flex-wrap items-center justify-between gap-4">
              <span className="grid size-12 place-items-center rounded-full border border-line bg-surface text-primary">
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
              <FileText
                aria-hidden="true"
                className="size-4 shrink-0 text-secondary"
              />
              <span>
                {sourceDirection
                  ? `Current focus: ${sourceDirection.title}.`
                  : "No current research direction is publicly listed."}
              </span>
            </div>
          </Surface>
        }
      />

      <section className="py-20 sm:py-24 lg:py-32">
        <Container>
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
            <div className="mt-12 grid gap-6 lg:grid-cols-[minmax(0,0.55fr)_minmax(0,1fr)] lg:gap-10">
              <Reveal delay={0.04}>
                <Surface className="h-full" padding="lg" variant="accent">
                  <p className="eyebrow text-primary">Review context</p>
                  <h3 className="mt-7 font-serif text-2xl leading-snug font-medium tracking-[-0.025em] text-ink sm:text-3xl">
                    {sourceDirection.title}
                  </h3>
                  <p className="mt-5 text-sm leading-7 text-muted sm:text-base">
                    {sourceDirection.summary}
                  </p>
                </Surface>
              </Reveal>

              <Reveal delay={0.08}>
                <Surface as="section" aria-labelledby="writing-questions-title" padding="none">
                  <div className="flex items-center gap-3 border-b border-line px-6 py-5 sm:px-8">
                    <CircleHelp
                      aria-hidden="true"
                      className="size-5 text-primary"
                      strokeWidth={1.7}
                    />
                    <h3
                      id="writing-questions-title"
                      className="text-sm font-semibold tracking-[-0.01em] text-ink"
                    >
                      Questions under review
                    </h3>
                  </div>
                  {sourceDirection.questions.length > 0 ? (
                    <ol className="divide-y divide-line px-6 sm:px-8">
                      {sourceDirection.questions.map((question, index) => (
                        <li
                          key={`${question}-${index}`}
                          className="grid gap-4 py-6 sm:grid-cols-[2.5rem_1fr] sm:items-start"
                        >
                          <span className="font-mono text-xs font-semibold text-primary">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <p className="text-sm leading-7 text-ink sm:text-base">
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
