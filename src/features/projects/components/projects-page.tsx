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
          <Surface
            as="aside"
            aria-label="Selected project overview"
            className="relative overflow-hidden shadow-lift"
            padding="lg"
            variant="accent"
          >
            <span
              aria-hidden="true"
              className="absolute -top-20 -right-16 size-48 rounded-full border border-primary/10 bg-primary/[0.055]"
            />
            <div className="relative flex items-center gap-3 text-primary">
              <span className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-canvas/70 shadow-sm">
                <BookOpen aria-hidden="true" className="size-5" />
              </span>
              <p className="eyebrow">{project.type}</p>
            </div>
            <p className="relative mt-5 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
              {project.indexSummaryTitle}
            </p>
            <p className="relative mt-4 text-sm leading-7 text-muted">
              {project.indexSummaryDescription}
            </p>
          </Surface>
        }
      />

      <section className="relative border-b border-line bg-surface/35 py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <article
              aria-labelledby="featured-project-title"
              className="relative overflow-hidden rounded-[1.75rem] border border-line/80 bg-canvas p-6 shadow-soft sm:p-8 lg:p-12"
            >
              <span
                aria-hidden="true"
                className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-primary to-secondary"
              />
              <span
                aria-hidden="true"
                className="absolute -top-28 -right-24 size-72 rounded-full bg-primary/[0.055] blur-3xl"
              />
              <div className="grid gap-10 lg:grid-cols-[minmax(0,0.38fr)_minmax(0,1fr)] lg:gap-16">
                <aside
                  aria-label={`${project.shortTitle} metadata`}
                  className="relative rounded-2xl border border-line/75 bg-surface/60 p-5 lg:sticky lg:top-32 lg:self-start lg:p-6"
                >
                  <div className="flex items-center justify-between gap-4">
                    <p className="eyebrow text-primary">01 / {project.type}</p>
                    <span
                      aria-hidden="true"
                      className="size-2 rounded-full bg-secondary ring-4 ring-secondary/10"
                    />
                  </div>
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

                <div className="relative">
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
                    <Surface
                      as="article"
                      className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary opacity-60 transition-opacity duration-300 group-hover:opacity-100 motion-reduce:transition-none"
                      />
                      <div className="flex items-center justify-between gap-4 border-b border-line pb-5">
                        <span className="eyebrow text-primary">
                          Area {String(index + 1).padStart(2, "0")}
                        </span>
                        <span
                          aria-hidden="true"
                          className="size-2 rounded-full bg-secondary transition-transform duration-300 group-hover:scale-150 motion-reduce:transform-none motion-reduce:transition-none"
                        />
                      </div>
                      <h3 className="mt-7 font-serif text-3xl leading-tight font-medium tracking-[-0.035em] transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none">
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
                className="group relative flex flex-col gap-6 overflow-hidden shadow-[0_24px_70px_-52px_rgb(20_25_35/0.45)] sm:flex-row sm:items-center sm:justify-between"
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
        <section className="foundation-grid relative border-b border-line bg-surface/25 py-18 sm:py-24 lg:py-28">
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
                    <Surface
                      as="article"
                      className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift focus-within:border-primary/50 motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-x-0 top-0 h-1 origin-left scale-x-50 bg-gradient-to-r from-primary to-secondary transition-transform duration-500 group-hover:scale-x-100 motion-reduce:transition-none"
                      />
                      <div className="flex items-center justify-between gap-4">
                        <Badge variant="accent">{additionalProject.type}</Badge>
                        <span className="font-mono text-xs font-semibold text-subtle">
                          {String(index + 2).padStart(2, "0")}
                        </span>
                      </div>
                      <h3 className="mt-6 font-serif text-3xl leading-tight font-medium tracking-[-0.035em] transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none">
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
