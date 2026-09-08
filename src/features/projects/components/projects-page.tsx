import { ArrowRight, ArrowUpRight, BookOpen, ExternalLink } from "lucide-react";

import { PageIntro } from "@/components/site/page-intro";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import type { ProjectsPageContent } from "@/features/projects/types";

export type ProjectsPageProps = {
  readonly content: ProjectsPageContent;
};

export function ProjectsPage({ content }: ProjectsPageProps) {
  const { projects } = content;
  const project = projects[0];
  const additionalProjects = projects.slice(1);

  if (!project) {
    return (
      <main id="main-content" className="flex-1 overflow-hidden">
        <PageIntro
          eyebrow="Selected academic work"
          title="No project work is listed yet."
          description="Academic and technical case studies will appear here when they are available for the portfolio."
          aside={
            <Surface as="aside" aria-label="Project status" padding="lg" variant="accent">
              <p className="eyebrow text-primary">Project status</p>
              <p className="mt-5 text-sm leading-7 text-muted">
                No project records are currently available.
              </p>
            </Surface>
          }
        />
      </main>
    );
  }

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <PageIntro
        eyebrow="Selected academic work"
        title="Projects shaped through learning and implementation."
        description="A focused view of academic work, presented through its scope, responsibilities, implementation, and technologies."
        actions={
          <>
            <ButtonLink href={project.href} size="lg">
              Read case study
              <ArrowRight aria-hidden="true" className="size-4" />
            </ButtonLink>
            {project.repository ? (
              <ButtonLink
                href={project.repository}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View project repository (opens in a new tab)"
                size="lg"
                variant="outline"
              >
                <ExternalLink aria-hidden="true" className="size-4" />
                View repository
              </ButtonLink>
            ) : null}
          </>
        }
        aside={
          <Surface as="aside" aria-label="Selected project overview" padding="lg" variant="accent">
            <div className="flex items-center gap-3 text-primary">
              <BookOpen aria-hidden="true" className="size-5" />
              <p className="eyebrow">{project.type}</p>
            </div>
            <p className="mt-5 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
              {project.indexSummaryTitle}
            </p>
            <p className="mt-4 text-sm leading-7 text-muted">
              {project.indexSummaryDescription}
            </p>
          </Surface>
        }
      />

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <article aria-labelledby="featured-project-title">
              <div className="grid gap-10 lg:grid-cols-[minmax(0,0.38fr)_minmax(0,1fr)] lg:gap-16">
                <aside
                  aria-label={`${project.shortTitle} metadata`}
                  className="border-t border-line pt-5 lg:sticky lg:top-32 lg:self-start"
                >
                  <p className="eyebrow text-primary">01 / {project.type}</p>
                  <dl className="mt-7 divide-y divide-line border-y border-line">
                    <div className="py-4">
                      <dt className="eyebrow text-subtle">Type</dt>
                      <dd className="mt-2 text-sm font-semibold leading-6 text-ink">
                        {project.type}
                      </dd>
                    </div>
                    {project.institution ? (
                      <div className="py-4">
                        <dt className="eyebrow text-subtle">Institution</dt>
                        <dd className="mt-2 text-sm font-semibold leading-6 text-ink">
                          {project.institution}
                        </dd>
                      </div>
                    ) : null}
                    {project.role ? (
                      <div className="py-4">
                        <dt className="eyebrow text-subtle">Role</dt>
                        <dd className="mt-2 text-sm font-semibold leading-6 text-ink">
                          {project.role}
                        </dd>
                      </div>
                    ) : null}
                    {project.advisor ? (
                      <div className="py-4">
                        <dt className="eyebrow text-subtle">Advisor</dt>
                        <dd className="mt-2 text-sm font-semibold leading-6 text-ink">
                          {project.advisor}
                        </dd>
                      </div>
                    ) : null}
                  </dl>
                </aside>

                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <Badge variant="accent">{project.type}</Badge>
                    {project.implementation ? (
                      <span className="eyebrow text-secondary">
                        {project.implementation}
                      </span>
                    ) : null}
                  </div>
                  <h2
                    id="featured-project-title"
                    className="mt-7 max-w-5xl text-balance font-serif text-4xl leading-[1.03] font-medium tracking-[-0.045em] sm:text-5xl lg:text-6xl"
                  >
                    {project.title}
                  </h2>
                  <p className="mt-7 max-w-3xl text-pretty text-base leading-8 text-muted sm:text-lg">
                    {project.summary}
                  </p>

                  <div className="mt-9 flex flex-wrap gap-3">
                    <ButtonLink href={project.href}>
                      Read case study
                      <ArrowRight aria-hidden="true" className="size-4" />
                    </ButtonLink>
                    {project.repository ? (
                      <ButtonLink
                        href={project.repository}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label="Open project repository (opens in a new tab)"
                        variant="outline"
                      >
                        Open repository
                        <ArrowUpRight aria-hidden="true" className="size-4" />
                      </ButtonLink>
                    ) : null}
                  </div>

                  {project.technologies.length > 0 ? (
                    <ul className="mt-10 flex flex-wrap gap-2" aria-label="Project technologies">
                      {project.technologies.map((technology, index) => (
                        <li key={`${index}-${technology}`}>
                          <Badge variant="outline">{technology}</Badge>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              </div>
            </article>
          </Reveal>
        </Container>
      </section>

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Selected contributions"
              title="Responsibilities across the project lifecycle."
              description="Only responsibilities documented in the public project record are presented here."
            />
          </Reveal>

          {project.contributions.length > 0 ? (
            <ol className="mt-12 grid gap-4 md:grid-cols-2" aria-label="Selected project contributions">
              {project.contributions.map((contribution, index) => (
                <li key={`${index}-${contribution.label}`}>
                  <Reveal className="h-full" delay={index * 0.05}>
                    <Surface as="article" className="h-full" padding="lg" variant="subtle">
                      <div className="flex items-center justify-between gap-4 border-b border-line pb-5">
                        <span className="eyebrow text-primary">
                          Area {String(index + 1).padStart(2, "0")}
                        </span>
                        <span aria-hidden="true" className="size-2 rounded-full bg-secondary" />
                      </div>
                      <h3 className="mt-7 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
                        {contribution.label}
                      </h3>
                      <p className="mt-4 text-sm leading-7 text-muted">
                        {contribution.description}
                      </p>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No public contribution records are listed for this project yet.
              </p>
            </Surface>
          )}

          {project.repository ? (
            <Reveal className="mt-10">
              <Surface
                as="aside"
                aria-label="Project repository"
                className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between"
                padding="lg"
                variant="raised"
              >
                <div>
                  <p className="eyebrow text-secondary">Project repository</p>
                  <p className="mt-3 max-w-2xl text-base leading-7 text-muted">
                    Explore the public repository and project context available
                    for this work.
                  </p>
                </div>
                <ButtonLink
                  href={project.repository}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Open project repository (opens in a new tab)"
                  variant="outline"
                >
                  Open repository
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              </Surface>
            </Reveal>
          ) : null}
        </Container>
      </section>

      {additionalProjects.length > 0 ? (
        <section className="border-b border-line py-18 sm:py-24 lg:py-28">
          <Container>
            <Reveal>
              <SectionHeading
                eyebrow="More project work"
                title="Additional case studies"
                description="Explore the other academic and technical projects currently available in the portfolio."
              />
            </Reveal>
            <ul className="mt-12 grid gap-5 md:grid-cols-2">
              {additionalProjects.map((additionalProject, index) => (
                <li key={additionalProject.slug}>
                  <Reveal className="h-full" delay={index * 0.05}>
                    <Surface as="article" className="h-full" padding="lg" variant="subtle">
                      <Badge variant="accent">{additionalProject.type}</Badge>
                      <h3 className="mt-6 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
                        {additionalProject.title}
                      </h3>
                      <p className="mt-5 text-sm leading-7 text-muted">
                        {additionalProject.summary}
                      </p>
                      <ButtonLink className="mt-7" href={additionalProject.href} variant="outline">
                        Read case study
                        <ArrowRight aria-hidden="true" className="size-4" />
                      </ButtonLink>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ul>
          </Container>
        </section>
      ) : null}
    </main>
  );
}
