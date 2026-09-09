import {
  ArrowUpRight,
  Award,
  BriefcaseBusiness,
  FileText,
  GraduationCap,
  Microscope,
} from "lucide-react";

import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import { ACADEMIC_CV_LINK } from "@/lib/academic-cv";

import type { ResumePageData } from "../types";

export type ResumePageProps = {
  readonly resume: ResumePageData;
};

export function ResumePage({ resume }: ResumePageProps) {
  return (
    <main id="main-content" className="overflow-hidden">
      <section className="foundation-grid relative isolate overflow-hidden border-b border-line/80 py-20 sm:py-28 lg:py-32">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-28 right-[-7rem] -z-10 size-[28rem] rounded-full bg-primary/[0.09] blur-3xl"
        />
        <Container size="content">
          <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-end lg:gap-16">
            <Reveal>
              <div>
              <Badge variant="accent">Academic + professional CV</Badge>
              <h1 className="mt-7 text-balance font-serif text-[clamp(3.5rem,8vw,7rem)] leading-[0.9] font-medium tracking-[-0.06em] text-ink">
                {resume.name}
              </h1>
              <p className="mt-7 max-w-3xl border-l-2 border-primary/30 pl-5 text-pretty text-lg leading-8 text-ink/75 sm:text-xl">
                {resume.headline}
              </p>
              <div className="mt-8 flex flex-wrap gap-2">
                {resume.positioning.map((position) => (
                  <Badge key={position} variant="outline">
                    {position}
                  </Badge>
                ))}
              </div>
              </div>
            </Reveal>

            <Reveal delay={0.08}>
            <Surface as="aside" className="relative overflow-hidden" variant="accent" padding="lg">
              <div
                aria-hidden="true"
                className="absolute -top-14 -right-14 size-40 rounded-full bg-primary/10 blur-2xl"
              />
              <span className="relative grid size-12 place-items-center rounded-xl border border-primary/20 bg-canvas/70 text-primary">
                <FileText aria-hidden="true" className="size-5" />
              </span>
              <h2 className="mt-5 text-xl font-semibold tracking-[-0.02em] text-ink">
                Current Academic CV
              </h2>
              <p className="mt-3 text-sm leading-6 text-muted">
                Open the current document directly for academic or professional
                review.
              </p>
              <ButtonLink className="mt-6" {...ACADEMIC_CV_LINK} wide>
                <FileText aria-hidden="true" className="size-4" />
                View Academic CV
                <ArrowUpRight aria-hidden="true" className="size-4" />
              </ButtonLink>
            </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      {resume.location || resume.emailAddress || resume.linkedInUrl ? (
        <section className="py-16 sm:py-20">
          <Container size="content">
            <Reveal>
            <div className="grid gap-px overflow-hidden rounded-2xl border border-line bg-line md:grid-cols-3">
            {resume.location ? (
              <div className="bg-canvas p-6 transition-colors duration-300 hover:bg-primary/[0.035] motion-reduce:transition-none sm:p-7">
                <p className="eyebrow text-primary">Location</p>
                <p className="mt-3 text-sm leading-6 text-ink/75">
                  {resume.location}
                </p>
              </div>
            ) : null}
            {resume.emailAddress ? (
              <div className="bg-canvas p-6 transition-colors duration-300 hover:bg-primary/[0.035] motion-reduce:transition-none sm:p-7">
                <p className="eyebrow text-primary">Email</p>
                <a
                  className="mt-3 block break-all rounded-sm text-sm leading-6 text-ink/75 underline decoration-line underline-offset-4 transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary motion-reduce:transition-none"
                  href={`mailto:${resume.emailAddress}`}
                >
                  {resume.emailAddress}
                </a>
              </div>
            ) : null}
            {resume.linkedInUrl ? (
              <div className="bg-canvas p-6 transition-colors duration-300 hover:bg-primary/[0.035] motion-reduce:transition-none sm:p-7">
                <p className="eyebrow text-primary">Professional profile</p>
                <a
                  className="mt-3 inline-flex items-center gap-2 rounded-sm text-sm leading-6 text-ink/75 underline decoration-line underline-offset-4 transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary motion-reduce:transition-none"
                  href={resume.linkedInUrl}
                  rel="noreferrer"
                  target="_blank"
                >
                  LinkedIn
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </a>
              </div>
            ) : null}
            </div>
            </Reveal>
          </Container>
        </section>
      ) : null}

      <section className="border-y border-line/80 bg-surface/45 py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <Reveal>
            <SectionHeading
              eyebrow="Professional experience"
              title="Data science and AI/ML engineering"
              description="Concise professional summaries prepared for academic and technical review."
            />
          </Reveal>
          {resume.experience.length > 0 ? (
            <ol className="relative mt-14 space-y-7 before:absolute before:top-6 before:bottom-6 before:left-[1.35rem] before:w-px before:bg-gradient-to-b before:from-primary/60 before:via-line-strong before:to-transparent before:content-[''] md:mt-16 md:before:left-[11.35rem]">
              {resume.experience.map((entry, index) => (
                <li
                  key={`${entry.organization}-${entry.role}-${entry.period ?? "undated"}`}
                  className="relative grid grid-cols-[2.75rem_minmax(0,1fr)] gap-4 md:grid-cols-[10rem_2.75rem_minmax(0,1fr)] md:gap-5"
                >
                  <div className="hidden pt-6 text-right md:block">
                    {entry.period ? (
                      <p className="font-mono text-xs font-semibold tracking-[0.08em] text-primary uppercase">
                        {entry.period}
                      </p>
                    ) : null}
                  </div>
                  <div className="relative z-10 flex justify-center pt-5">
                    <span className="grid size-11 place-items-center rounded-full border border-primary/35 bg-canvas text-primary shadow-[0_0_0_7px_var(--color-surface)]">
                      <BriefcaseBusiness aria-hidden="true" className="size-4" />
                    </span>
                  </div>
                  <Reveal delay={Math.min(index * 0.05, 0.15)}>
                    <Surface
                      as="article"
                      className="group relative overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary/45"
                      />
                      <div className="flex flex-wrap items-center justify-between gap-4">
                        <Badge variant={entry.isCurrent ? "accent" : "neutral"}>
                          {entry.isCurrent ? "Current" : "Previous"}
                        </Badge>
                        {entry.period ? (
                          <p className="font-mono text-xs tracking-[0.08em] text-muted uppercase md:hidden">
                            {entry.period}
                          </p>
                        ) : null}
                      </div>
                      <h3 className="mt-7 font-serif text-3xl leading-tight font-medium tracking-[-0.04em] text-ink transition-colors duration-300 group-hover:text-primary motion-reduce:transition-none">
                        {entry.role}
                      </h3>
                      <p className="mt-3 text-sm font-semibold leading-6 text-primary">
                        {entry.engagement}: {entry.organization}
                        {entry.location ? ` · ${entry.location}` : null}
                      </p>
                      <p className="mt-6 max-w-3xl text-base leading-8 text-ink/70">
                        {entry.summary}
                      </p>
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ol>
          ) : (
            <Surface className="mt-10" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No professional experience records are currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <div className="grid gap-16 lg:grid-cols-2 lg:gap-20">
            <div>
              <SectionHeading
                eyebrow="Education"
                title="Academic background"
                size="sm"
              />
              {resume.education.length > 0 ? (
                <ol className="mt-8 space-y-4">
                  {resume.education.map((entry) => (
                    <li key={`${entry.institution}-${entry.degree}-${entry.period ?? "undated"}`}>
                      <Surface
                        as="article"
                        className="group grid grid-cols-[3rem_minmax(0,1fr)] gap-4 transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:bg-primary/[0.025] hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                        variant="subtle"
                      >
                        <span className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-canvas text-primary transition-colors duration-300 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none">
                          <GraduationCap aria-hidden="true" className="size-5" />
                        </span>
                        <div>
                        <h3 className="font-serif text-xl leading-tight font-medium tracking-[-0.025em] text-ink">
                          {entry.degree}
                        </h3>
                        <p className="mt-2 text-sm leading-6 text-muted">
                          {entry.institution}
                        </p>
                        {entry.period ? (
                          <p className="mt-4 font-mono text-xs tracking-[0.08em] text-primary uppercase">
                            {entry.period}
                          </p>
                        ) : null}
                        </div>
                      </Surface>
                    </li>
                  ))}
                </ol>
              ) : (
                <Surface className="mt-8" variant="subtle">
                  <p className="text-sm leading-7 text-muted">
                    No education records are currently published.
                  </p>
                </Surface>
              )}
            </div>

            <div>
              {resume.research ? (
                <>
                  <SectionHeading
                    eyebrow="Research direction"
                    title={resume.research.format ?? resume.research.title}
                    description={resume.research.summary}
                    size="sm"
                  />
                  <Surface className="relative mt-8 overflow-hidden" variant="accent">
                    <div
                      aria-hidden="true"
                      className="absolute -top-14 -right-14 size-36 rounded-full bg-primary/10 blur-2xl"
                    />
                    <div className="flex items-center gap-3">
                      <Microscope aria-hidden="true" className="size-5 text-primary" />
                      <Badge variant="accent">{resume.research.stage}</Badge>
                    </div>
                    {resume.research.interests.length > 0 ? (
                      <>
                        <h3 className="mt-6 text-sm font-semibold text-ink">
                          Research interests
                        </h3>
                        <ul className="mt-4 space-y-3">
                          {resume.research.interests.map((interest) => (
                            <li key={interest} className="flex gap-3 text-sm leading-6 text-muted">
                              <span
                                aria-hidden="true"
                                className="mt-2 size-1.5 shrink-0 rounded-full bg-primary"
                              />
                              {interest}
                            </li>
                          ))}
                        </ul>
                      </>
                    ) : null}
                  </Surface>
                </>
              ) : (
                <>
                  <SectionHeading
                    eyebrow="Research direction"
                    title="No research direction is currently published."
                    size="sm"
                  />
                  <Surface className="mt-8" variant="subtle">
                    <p className="text-sm leading-7 text-muted">
                      Published research details will appear here.
                    </p>
                  </Surface>
                </>
              )}
            </div>
          </div>
        </Container>
      </section>

      {resume.project ? (
        <section className="border-y border-line/80 bg-surface/45 py-20 sm:py-28 lg:py-32">
          <Container size="content">
            <SectionHeading
              eyebrow="Selected academic project"
              title={resume.project.title}
              description={resume.project.summary}
            />
            <Surface
              className="group relative mt-10 overflow-hidden transition-[border-color,box-shadow] duration-300 hover:border-primary/35 hover:shadow-lift motion-reduce:transition-none"
              padding="lg"
            >
              <span
                aria-hidden="true"
                className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-secondary to-transparent"
              />
              <dl className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                {resume.project.type ? (
                  <div>
                    <dt className="eyebrow text-muted">Type</dt>
                    <dd className="mt-2 text-sm leading-6 text-ink">
                      {resume.project.type}
                    </dd>
                  </div>
                ) : null}
                {resume.project.institution ? (
                  <div>
                    <dt className="eyebrow text-muted">Institution</dt>
                    <dd className="mt-2 text-sm leading-6 text-ink">
                      {resume.project.institution}
                    </dd>
                  </div>
                ) : null}
                {resume.project.role ? (
                  <div>
                    <dt className="eyebrow text-muted">Role</dt>
                    <dd className="mt-2 text-sm leading-6 text-ink">
                      {resume.project.role}
                    </dd>
                  </div>
                ) : null}
                {resume.project.advisor ? (
                  <div>
                    <dt className="eyebrow text-muted">Advisor</dt>
                    <dd className="mt-2 text-sm leading-6 text-ink">
                      {resume.project.advisor}
                    </dd>
                  </div>
                ) : null}
              </dl>
              {resume.project.technologies.length > 0 ? (
                <ul className="mt-8 flex flex-wrap gap-2" aria-label="Project technologies">
                  {resume.project.technologies.map((technology) => (
                    <li key={technology}>
                      <Badge variant="outline">{technology}</Badge>
                    </li>
                  ))}
                </ul>
              ) : null}
              {resume.project.repositoryUrl ? (
                <a
                  className="mt-8 inline-flex items-center gap-2 text-sm font-semibold text-primary underline decoration-primary/30 underline-offset-4"
                  href={resume.project.repositoryUrl}
                  rel="noreferrer"
                  target="_blank"
                >
                  View project repository
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </a>
              ) : null}
            </Surface>
          </Container>
        </section>
      ) : null}

      {resume.certifications.length > 0 ? (
        <section className="py-20 sm:py-28 lg:py-32">
          <Container size="content">
            <SectionHeading
              eyebrow="Certifications"
              title="Verified credentials"
              description="Only certifications currently published in the portfolio are shown."
            />
            <ul className="mt-12 grid gap-4 md:grid-cols-2">
              {resume.certifications.map((certification) => (
                <li key={`${certification.name}-${certification.issuer}`}>
                  <Surface
                    as="article"
                    className="group h-full transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:bg-primary/[0.025] hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                    variant="subtle"
                  >
                    <span className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-canvas text-primary transition-colors duration-300 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none">
                      <Award aria-hidden="true" className="size-5" />
                    </span>
                    <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink">
                      {certification.name}
                    </h3>
                    <p className="mt-2 text-sm text-muted">{certification.issuer}</p>
                    {certification.period ? (
                      <p className="mt-4 font-mono text-xs tracking-[0.08em] text-primary uppercase">
                        {certification.period}
                      </p>
                    ) : null}
                    {certification.credentialUrl ? (
                      <a
                        className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-primary underline decoration-primary/30 underline-offset-4"
                        href={certification.credentialUrl}
                        rel="noreferrer"
                        target="_blank"
                      >
                        View credential
                        <ArrowUpRight aria-hidden="true" className="size-4" />
                      </a>
                    ) : null}
                  </Surface>
                </li>
              ))}
            </ul>
          </Container>
        </section>
      ) : null}

      <section className="border-t border-line/80 bg-surface/35 py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <SectionHeading
            eyebrow="Technical expertise"
            title="Skills and platforms"
            description="The categorized technical areas represented in my academic CV."
          />
          {resume.expertise.length > 0 ? (
            <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {resume.expertise.map((group, index) => (
                <Surface
                  key={group.category}
                  as="article"
                  className="group transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                  variant="subtle"
                >
                  <div className="flex items-center justify-between gap-4 border-b border-line pb-5">
                    <span className="grid size-9 place-items-center rounded-full border border-primary/20 bg-canvas font-mono text-[0.65rem] font-semibold text-primary">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="eyebrow text-subtle">Skill group</span>
                  </div>
                  <h3 className="mt-6 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink">
                    {group.category}
                  </h3>
                  <ul className="mt-6 flex flex-wrap gap-2" aria-label={group.category}>
                    {group.skills.map((skill) => (
                      <li key={skill}>
                        <Badge variant="outline">{skill}</Badge>
                      </li>
                    ))}
                  </ul>
                </Surface>
              ))}
            </div>
          ) : (
            <Surface className="mt-10" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No technical skill records are currently published.
              </p>
            </Surface>
          )}
        </Container>
      </section>
    </main>
  );
}
