import { ArrowLeft, ArrowUpRight, ExternalLink } from "lucide-react";

import { DetailIntro } from "@/components/site/detail-intro";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import type { ProjectDetailRecord } from "@/features/projects/types";

export type ProjectDetailPageProps = {
  readonly project: ProjectDetailRecord;
};

export function ProjectDetailPage({ project }: ProjectDetailPageProps) {
  const metadata = [
    { label: "Type", value: project.type },
    ...(project.institution
      ? [{ label: "Institution", value: project.institution }]
      : []),
    ...(project.advisor
      ? [{ label: "Advisor", value: project.advisor }]
      : []),
    ...(project.role ? [{ label: "Role", value: project.role }] : []),
    ...(project.implementation
      ? [{ label: "Format", value: project.implementation }]
      : []),
  ];

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <DetailIntro
        eyebrow={project.type}
        currentLabel={project.shortTitle}
        title={project.title}
        description={project.summary}
        parentHref="/projects"
        parentLabel="Projects"
        metadata={metadata}
        actions={
          <>
            {project.repository ? (
              <ButtonLink
                href={project.repository}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View project repository (opens in a new tab)"
                size="lg"
              >
                <ExternalLink aria-hidden="true" className="size-4" />
                View repository
                <ArrowUpRight aria-hidden="true" className="size-4" />
              </ButtonLink>
            ) : null}
            <ButtonLink href="/projects" size="lg" variant="outline">
              <ArrowLeft aria-hidden="true" className="size-4" />
              All projects
            </ButtonLink>
          </>
        }
      />

      <section className="relative border-b border-line bg-surface/35 py-18 sm:py-24 lg:py-28">
        <Container>
          <div className="grid gap-10 lg:grid-cols-[minmax(0,0.38fr)_minmax(0,1fr)] lg:gap-16">
            <Reveal>
              <SectionHeading
                eyebrow="Project overview"
                title={project.caseStudy.overviewTitle}
                size="sm"
              />
            </Reveal>
            <Reveal delay={0.06}>
              <Surface
                as="article"
                className="relative overflow-hidden shadow-lift"
                padding="lg"
                variant="raised"
              >
                <span
                  aria-hidden="true"
                  className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary"
                />
                <span
                  aria-hidden="true"
                  className="absolute -top-20 -right-16 size-52 rounded-full bg-primary/[0.055] blur-3xl"
                />
                <h3 className="relative font-serif text-3xl leading-tight font-medium tracking-[-0.035em] sm:text-4xl">
                  Overview
                </h3>
                <p className="relative mt-5 text-base leading-8 text-muted sm:text-lg">
                  {project.summary}
                </p>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Project features"
              title={project.caseStudy.learningTitle}
              description={project.caseStudy.learningDescription}
            />
          </Reveal>

          {project.learningFeatures.length > 0 ? (
            <ul
              className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
              aria-label="Project features"
            >
              {project.learningFeatures.map((feature, index) => (
                <li key={`${index}-${feature}`}>
                  <Reveal className="h-full" delay={index * 0.04}>
                    <Surface
                      as="article"
                      className="group relative h-full min-h-52 overflow-hidden transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-x-0 top-0 h-1 origin-left scale-x-50 bg-gradient-to-r from-primary to-secondary transition-transform duration-500 group-hover:scale-x-100 motion-reduce:transition-none"
                      />
                      <span
                        aria-hidden="true"
                        className="absolute -right-2 -bottom-8 font-serif text-[7rem] leading-none font-medium tracking-[-0.08em] text-primary/[0.045] transition-transform duration-500 group-hover:-translate-x-2 group-hover:-translate-y-2 motion-reduce:transform-none motion-reduce:transition-none"
                      >
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <p className="relative eyebrow text-primary">
                        Feature {String(index + 1).padStart(2, "0")}
                      </p>
                      <h3 className="relative mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] transition-colors duration-200 group-hover:text-primary motion-reduce:transition-none">
                        {feature}
                      </h3>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ul>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No public feature records are listed for this project yet.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="border-b border-line bg-surface/20 py-18 sm:py-24 lg:py-28">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Project contributions"
              title={project.caseStudy.contributionsTitle}
              description={project.caseStudy.contributionsDescription}
            />
          </Reveal>

          {project.contributions.length > 0 ? (
            <ol
              className="mt-12 grid gap-4 md:grid-cols-2"
              aria-label="Project contributions"
            >
              {project.contributions.map((contribution, index) => (
                <li key={`${index}-${contribution.label}`}>
                  <Reveal className="h-full" delay={index * 0.05}>
                    <Surface
                      as="article"
                      className="group relative h-full overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-secondary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="raised"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-secondary to-primary opacity-65 transition-opacity duration-300 group-hover:opacity-100 motion-reduce:transition-none"
                      />
                      <p className="eyebrow text-secondary">
                        Contribution {String(index + 1).padStart(2, "0")}
                      </p>
                      <h3 className="mt-6 font-serif text-3xl leading-tight font-medium tracking-[-0.035em] transition-colors duration-200 group-hover:text-secondary motion-reduce:transition-none">
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
        </Container>
      </section>

      <section className="foundation-grid relative border-b border-line py-18 sm:py-24 lg:py-28">
        <Container>
          <div className="grid gap-10 lg:grid-cols-[minmax(0,0.42fr)_minmax(0,1fr)] lg:gap-16">
            <Reveal>
              <SectionHeading
                eyebrow="Technology"
                title={project.caseStudy.technologyTitle}
                description={project.caseStudy.technologyDescription}
                size="sm"
              />
            </Reveal>
            <Reveal delay={0.06}>
              <Surface
                as="article"
                className="relative overflow-hidden shadow-[0_24px_70px_-52px_rgb(20_25_35/0.45)]"
                padding="lg"
                variant="accent"
              >
                <span
                  aria-hidden="true"
                  className="absolute -top-24 -right-20 size-56 rounded-full border border-primary/10 bg-primary/[0.04]"
                />
                <h3 className="relative font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
                  Technologies
                </h3>
                {project.technologies.length > 0 ? (
                  <ul
                    className="relative mt-7 flex flex-wrap gap-2"
                    aria-label="Project technologies"
                  >
                    {project.technologies.map((technology, index) => (
                      <li key={`${index}-${technology}`}>
                        <Badge variant="outline">{technology}</Badge>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-7 text-sm leading-7 text-muted">
                    No public technology records are listed yet.
                  </p>
                )}
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      {project.repository ? (
        <section className="border-b border-line bg-surface/30 py-18 sm:py-24 lg:py-28">
          <Container>
            <Reveal>
              <Surface
                as="article"
                className="relative flex flex-col gap-7 overflow-hidden shadow-lift lg:flex-row lg:items-end lg:justify-between"
                padding="lg"
                variant="raised"
              >
                <div className="max-w-3xl">
                  <p className="eyebrow text-primary">Project repository</p>
                  <h2 className="mt-5 font-serif text-4xl leading-tight font-medium tracking-[-0.04em] sm:text-5xl">
                    Explore the public project repository.
                  </h2>
                  <p className="mt-5 text-base leading-8 text-muted">
                    Visit the repository associated with this project.
                  </p>
                </div>
                <ButtonLink
                  href={project.repository}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Open project repository (opens in a new tab)"
                  size="lg"
                >
                  Open repository
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              </Surface>
            </Reveal>
          </Container>
        </section>
      ) : null}
    </main>
  );
}
