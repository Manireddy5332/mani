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

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
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
              <Surface as="article" padding="lg" variant="raised">
                <h3 className="font-serif text-3xl leading-tight font-medium tracking-[-0.035em] sm:text-4xl">
                  Overview
                </h3>
                <p className="mt-5 text-base leading-8 text-muted sm:text-lg">
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
                    <Surface as="article" className="h-full" padding="lg" variant="subtle">
                      <p className="eyebrow text-primary">
                        Feature {String(index + 1).padStart(2, "0")}
                      </p>
                      <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em]">
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

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
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
                    <Surface as="article" className="h-full" padding="lg" variant="raised">
                      <p className="eyebrow text-secondary">
                        Contribution {String(index + 1).padStart(2, "0")}
                      </p>
                      <h3 className="mt-6 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
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

      <section className="border-b border-line py-18 sm:py-24 lg:py-28">
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
              <Surface as="article" padding="lg" variant="accent">
                <h3 className="font-serif text-3xl leading-tight font-medium tracking-[-0.035em]">
                  Technologies
                </h3>
                {project.technologies.length > 0 ? (
                  <ul
                    className="mt-7 flex flex-wrap gap-2"
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
        <section className="border-b border-line py-18 sm:py-24 lg:py-28">
          <Container>
            <Reveal>
              <Surface
                as="article"
                className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between"
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
