import {
  ArrowRight,
  Award,
  BriefcaseBusiness,
  GraduationCap,
  MapPin,
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
import { ProfilePhoto } from "@/components/site/profile-photo";
import { ACADEMIC_CV_LINK } from "@/lib/academic-cv";

import type { ProfilePageData } from "../types";

export type AboutPageProps = {
  readonly profile: ProfilePageData;
};

export function AboutPage({ profile }: AboutPageProps) {
  const currentExperience =
    profile.experience.find((entry) => entry.isCurrent) ?? profile.experience[0];

  return (
    <main id="main-content" className="overflow-hidden">
      <section className="foundation-grid relative isolate overflow-hidden border-b border-line/80 py-20 sm:py-28 lg:py-32">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-32 right-[-8rem] -z-10 size-[30rem] rounded-full bg-primary/[0.09] blur-3xl"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute bottom-[-12rem] left-[28%] -z-10 size-96 rounded-full bg-secondary/[0.07] blur-3xl"
        />
        <Container size="content">
          <Reveal>
            <div className="max-w-5xl">
              <div className="flex flex-wrap items-center gap-4">
                <Badge variant="accent">About</Badge>
                <span className="hidden h-px w-16 bg-gradient-to-r from-primary/60 to-transparent sm:block" />
                <span className="font-mono text-[0.65rem] tracking-[0.14em] text-subtle uppercase">
                  {profile.name}
                </span>
              </div>
              <h1 className="mt-7 text-balance font-serif text-[clamp(3.25rem,7vw,6.5rem)] leading-[0.92] font-medium tracking-[-0.055em] text-ink">
                Professional practice, shaped by academic inquiry.
              </h1>
              <p className="mt-8 max-w-3xl border-l-2 border-primary/30 pl-5 text-pretty text-lg leading-8 text-ink/75 sm:text-xl sm:leading-9">
                {profile.introduction}
              </p>
              <div className="mt-8 flex flex-wrap gap-2">
                {profile.positioning.map((position) => (
                  <Badge key={position} variant="outline">
                    {position}
                  </Badge>
                ))}
              </div>
              <div className="mt-10 flex flex-wrap gap-3">
                <ButtonLink {...ACADEMIC_CV_LINK}>
                  View Academic CV
                  <ArrowRight aria-hidden="true" className="size-4" />
                </ButtonLink>
                <ButtonLink href="/contact" variant="secondary">
                  Contact me
                </ButtonLink>
              </div>
            </div>
          </Reveal>
        </Container>
      </section>

      <section className="py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <div className="grid gap-12 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)] lg:gap-16">
            <div>
              <Reveal>
                <SectionHeading
                  eyebrow="Profile"
                  title="A path from data science toward advanced AI/ML work and research."
                  description={profile.headline}
                />
              </Reveal>
              {profile.progression.length > 0 ? (
                <div className="mt-10 space-y-1">
                  {profile.progression.map((paragraph, index) => (
                    <Reveal key={paragraph} delay={Math.min(index * 0.05, 0.15)}>
                      <div className="group grid gap-4 border-t border-line py-6 sm:grid-cols-[3rem_minmax(0,1fr)]">
                        <span className="font-mono text-xs font-semibold tracking-[0.12em] text-primary">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <p className="max-w-3xl text-base leading-8 text-ink/75 transition-colors duration-300 group-hover:text-ink motion-reduce:transition-none sm:text-lg">
                          {paragraph}
                        </p>
                      </div>
                    </Reveal>
                  ))}
                </div>
              ) : null}
            </div>

            <Reveal delay={0.08}>
              <Surface
                as="aside"
                variant="accent"
                padding="lg"
                className="relative h-fit overflow-hidden lg:sticky lg:top-28"
              >
              <div
                aria-hidden="true"
                className="absolute -top-16 -right-16 size-40 rounded-full bg-primary/10 blur-2xl"
              />
              <p className="eyebrow relative text-primary">At a glance</p>
              <ProfilePhoto photo={profile.photo} />
              <dl className="relative mt-7 divide-y divide-line">
                {profile.location ? (
                  <div className="py-5 first:pt-0 last:pb-0">
                    <dt className="flex items-center gap-3 text-sm font-semibold text-ink">
                      <span className="grid size-9 place-items-center rounded-full border border-primary/20 bg-canvas/65 text-primary">
                        <MapPin aria-hidden="true" className="size-4" />
                      </span>
                      Location
                    </dt>
                    <dd className="mt-3 pl-12 text-sm leading-6 text-muted">
                      {profile.location}
                    </dd>
                  </div>
                ) : null}
                {currentExperience ? (
                  <div className="py-5 first:pt-0 last:pb-0">
                    <dt className="flex items-center gap-3 text-sm font-semibold text-ink">
                      <span className="grid size-9 place-items-center rounded-full border border-primary/20 bg-canvas/65 text-primary">
                        <BriefcaseBusiness aria-hidden="true" className="size-4" />
                      </span>
                      {currentExperience.isCurrent ? "Current role" : "Professional role"}
                    </dt>
                    <dd className="mt-3 pl-12 text-sm leading-6 text-muted">
                      {currentExperience.role}
                      <br />
                      {currentExperience.engagement}: {currentExperience.organization}
                    </dd>
                  </div>
                ) : null}
                {profile.research ? (
                  <div className="py-5 first:pt-0 last:pb-0">
                    <dt className="flex items-center gap-3 text-sm font-semibold text-ink">
                      <span className="grid size-9 place-items-center rounded-full border border-primary/20 bg-canvas/65 text-primary">
                        <Microscope aria-hidden="true" className="size-4" />
                      </span>
                      Research status
                    </dt>
                    <dd className="mt-3 pl-12 text-sm leading-6 text-muted">
                      {profile.research.stage}
                      {profile.research.format ? ` · ${profile.research.format}` : null}
                    </dd>
                  </div>
                ) : null}
                {!profile.location && !currentExperience && !profile.research ? (
                  <div>
                    <dt className="sr-only">Profile details</dt>
                    <dd className="text-sm leading-6 text-muted">
                      Additional profile details are not currently published.
                    </dd>
                  </div>
                ) : null}
              </dl>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section
        id="experience"
        className="scroll-mt-32 border-y border-line/80 bg-surface/45 py-20 sm:py-28 lg:py-32"
      >
        <Container size="content">
          <Reveal>
            <SectionHeading
              eyebrow="Professional progression"
              title="Applied work across data science and AI/ML engineering."
              description="A concise public view of verified professional experience."
            />
          </Reveal>
          {profile.experience.length > 0 ? (
            <ol className="relative mt-14 space-y-7 before:absolute before:top-6 before:bottom-6 before:left-[1.35rem] before:w-px before:bg-gradient-to-b before:from-primary/60 before:via-line-strong before:to-transparent before:content-[''] md:mt-16 md:before:left-[10.35rem]">
              {profile.experience.map((entry, index) => (
                <li
                  key={`${entry.organization}-${entry.role}-${entry.period ?? "undated"}`}
                  className="relative grid grid-cols-[2.75rem_minmax(0,1fr)] gap-4 md:grid-cols-[9rem_2.75rem_minmax(0,1fr)] md:gap-5"
                >
                  <div className="hidden pt-6 text-right md:block">
                    {entry.period ? (
                      <p className="font-mono text-xs font-semibold tracking-[0.08em] text-primary uppercase">
                        {entry.period}
                      </p>
                    ) : null}
                  </div>
                  <div className="relative z-10 flex justify-center pt-5">
                    <span className="grid size-11 place-items-center rounded-full border border-primary/35 bg-canvas shadow-[0_0_0_7px_var(--color-surface)]">
                      <span aria-hidden="true" className="size-2.5 rounded-full bg-primary" />
                      <span className="sr-only">Timeline point {index + 1}</span>
                    </span>
                  </div>
                  <Reveal delay={Math.min(index * 0.05, 0.15)}>
                    <Surface
                      as="article"
                      padding="lg"
                      className="group relative overflow-hidden transition-[border-color,transform,box-shadow] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary/45"
                      />
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <Badge variant={entry.isCurrent ? "accent" : "neutral"}>
                          {entry.isCurrent ? "Current" : "Previous"}
                        </Badge>
                        {entry.period ? (
                          <span className="font-mono text-xs tracking-[0.08em] text-muted uppercase md:hidden">
                            {entry.period}
                          </span>
                        ) : null}
                      </div>
                      <h3 className="mt-7 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.04em] text-ink transition-colors duration-300 group-hover:text-primary motion-reduce:transition-none">
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

      <section id="education" className="scroll-mt-32 py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <div className="grid gap-16 lg:grid-cols-2 lg:gap-20">
            <div>
              <Reveal>
                {profile.research ? (
                  <>
                    <SectionHeading
                      eyebrow="Research direction"
                      title={profile.research.title}
                      description={profile.research.summary}
                      size="sm"
                    />
                  {profile.research.interests.length > 0 ? (
                    <ul className="mt-9 space-y-3">
                      {profile.research.interests.map((interest, index) => (
                        <li
                          key={interest}
                          className="group grid grid-cols-[2.5rem_minmax(0,1fr)] items-start gap-3 rounded-xl border border-line/70 bg-ink/[0.02] p-4 text-sm leading-6 text-ink/75 transition-[border-color,transform,background-color] duration-300 hover:translate-x-1 hover:border-primary/30 hover:bg-primary/[0.04] motion-reduce:transform-none motion-reduce:transition-none"
                        >
                          <span className="font-mono text-[0.65rem] font-semibold tracking-[0.1em] text-primary">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          {interest}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                    <ButtonLink className="mt-8" href={profile.research.href} variant="outline">
                      Explore research
                      <ArrowRight aria-hidden="true" className="size-4" />
                    </ButtonLink>
                  </>
                ) : (
                  <>
                    <SectionHeading
                      eyebrow="Research direction"
                      title="Research details are not currently published."
                      size="sm"
                    />
                    <Surface className="mt-8" variant="subtle">
                      <p className="text-sm leading-7 text-muted">
                        Published research interests and directions will appear here.
                      </p>
                    </Surface>
                  </>
                )}
              </Reveal>
            </div>

            <div>
              <Reveal>
                <SectionHeading
                  eyebrow="Education"
                  title="Academic foundation"
                  size="sm"
                />
              </Reveal>
              {profile.education.length > 0 ? (
                <ol className="mt-9 space-y-4">
                  {profile.education.map((entry, index) => (
                    <li key={`${entry.institution}-${entry.degree}-${entry.period ?? "undated"}`}>
                      <Reveal delay={Math.min(index * 0.05, 0.15)}>
                        <Surface
                          as="article"
                          variant="subtle"
                          className="group grid grid-cols-[3rem_minmax(0,1fr)] gap-4 transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/35 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                        >
                          <span className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-primary/[0.06] text-primary transition-colors duration-300 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none">
                            <GraduationCap aria-hidden="true" className="size-5" />
                          </span>
                          <div>
                            <p className="font-mono text-[0.65rem] font-semibold tracking-[0.12em] text-subtle uppercase">
                              Record {String(index + 1).padStart(2, "0")}
                            </p>
                            <h3 className="mt-3 font-serif text-xl leading-tight font-medium tracking-[-0.025em] text-ink">
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
                      </Reveal>
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
          </div>
        </Container>
      </section>

      {profile.certifications.length > 0 ? (
        <section className="border-t border-line/80 bg-surface/35 py-20 sm:py-28 lg:py-32">
          <Container size="content">
            <Reveal>
              <SectionHeading
                eyebrow="Certifications"
                title="Verified credentials"
                description="Only certifications currently published in the portfolio are shown."
              />
            </Reveal>
            <ul className="mt-14 grid gap-5 md:grid-cols-2">
              {profile.certifications.map((certification, index) => (
                <li key={`${certification.name}-${certification.issuer}`}>
                  <Reveal className="h-full" delay={Math.min(index * 0.05, 0.15)}>
                    <Surface
                      as="article"
                      className="group h-full transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-canvas hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
                      variant="subtle"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <span className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-primary/[0.06] text-primary transition-colors duration-300 group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none">
                          <Award aria-hidden="true" className="size-5" />
                        </span>
                        <span className="font-mono text-[0.65rem] tracking-[0.12em] text-subtle uppercase">
                          Verified
                        </span>
                      </div>
                      <h3 className="mt-7 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink">
                        {certification.name}
                      </h3>
                      <p className="mt-3 text-sm text-muted">{certification.issuer}</p>
                      {certification.period ? (
                        <p className="mt-4 font-mono text-xs tracking-[0.08em] text-primary uppercase">
                          {certification.period}
                        </p>
                      ) : null}
                      {certification.credentialUrl ? (
                        <a
                          className="mt-6 inline-flex items-center gap-2 rounded-sm text-sm font-semibold text-primary underline decoration-primary/30 underline-offset-4 transition-colors hover:text-primary-hover focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary motion-reduce:transition-none"
                          href={certification.credentialUrl}
                          rel="noreferrer"
                          target="_blank"
                        >
                          View credential
                          <ArrowRight aria-hidden="true" className="size-4" />
                        </a>
                      ) : null}
                    </Surface>
                  </Reveal>
                </li>
              ))}
            </ul>
          </Container>
        </section>
      ) : null}

      <section id="expertise" className="scroll-mt-32 border-t border-line/80 py-20 sm:py-28 lg:py-32">
        <Container size="content">
          <Reveal>
            <SectionHeading
              eyebrow="Technical expertise"
              title="A categorized view of tools and technical foundations."
              description="Organized for clarity, without reducing the portfolio to a wall of technology logos."
            />
          </Reveal>
          {profile.expertise.length > 0 ? (
            <div className="mt-14 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
              {profile.expertise.map((group, index) => (
                <Reveal key={group.category} className="h-full" delay={(index % 3) * 0.05}>
                  <Surface
                    as="article"
                    className="group h-full transition-[border-color,transform,box-shadow,background-color] duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-primary/[0.025] hover:shadow-lift motion-reduce:transform-none motion-reduce:transition-none"
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
                </Reveal>
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
