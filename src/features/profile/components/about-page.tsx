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
  SectionHeading,
  Surface,
} from "@/components/ui";

import type { ProfilePageData } from "../types";

export type AboutPageProps = {
  readonly profile: ProfilePageData;
};

export function AboutPage({ profile }: AboutPageProps) {
  const currentExperience =
    profile.experience.find((entry) => entry.isCurrent) ?? profile.experience[0];

  return (
    <main id="main-content">
      <section className="foundation-grid border-b border-line/80 py-20 sm:py-24 lg:py-28">
        <Container size="content">
          <div className="max-w-4xl">
            <Badge variant="accent">About</Badge>
            <h1 className="mt-7 text-balance font-serif text-5xl leading-[0.98] font-medium tracking-[-0.045em] text-ink sm:text-6xl lg:text-7xl">
              Professional practice, shaped by academic inquiry.
            </h1>
            <p className="mt-7 max-w-3xl text-pretty text-lg leading-8 text-ink/75 sm:text-xl sm:leading-9">
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
              <ButtonLink href="/resume">
                View CV overview
                <ArrowRight aria-hidden="true" className="size-4" />
              </ButtonLink>
              <ButtonLink href="/contact" variant="secondary">
                Contact me
              </ButtonLink>
            </div>
          </div>
        </Container>
      </section>

      <section className="py-20 sm:py-24">
        <Container size="content">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1.35fr)_minmax(17rem,0.65fr)] lg:gap-14">
            <div>
              <SectionHeading
                eyebrow="Profile"
                title="A path from data science toward advanced AI/ML work and research."
                description={profile.headline}
              />
              {profile.progression.length > 0 ? (
                <div className="mt-9 space-y-5 text-base leading-8 text-ink/75 sm:text-lg">
                  {profile.progression.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </div>
              ) : null}
            </div>

            <Surface as="aside" variant="accent" padding="lg" className="h-fit">
              <p className="eyebrow text-primary">At a glance</p>
              <dl className="mt-7 space-y-6">
                {profile.location ? (
                  <div>
                    <dt className="flex items-center gap-2 text-sm font-semibold text-ink">
                      <MapPin aria-hidden="true" className="size-4 text-primary" />
                      Location
                    </dt>
                    <dd className="mt-2 text-sm leading-6 text-muted">
                      {profile.location}
                    </dd>
                  </div>
                ) : null}
                {currentExperience ? (
                  <div>
                    <dt className="flex items-center gap-2 text-sm font-semibold text-ink">
                      <BriefcaseBusiness
                        aria-hidden="true"
                        className="size-4 text-primary"
                      />
                      {currentExperience.isCurrent ? "Current role" : "Professional role"}
                    </dt>
                    <dd className="mt-2 text-sm leading-6 text-muted">
                      {currentExperience.role}
                      <br />
                      {currentExperience.engagement}: {currentExperience.organization}
                    </dd>
                  </div>
                ) : null}
                {profile.research ? (
                  <div>
                    <dt className="flex items-center gap-2 text-sm font-semibold text-ink">
                      <Microscope
                        aria-hidden="true"
                        className="size-4 text-primary"
                      />
                      Research status
                    </dt>
                    <dd className="mt-2 text-sm leading-6 text-muted">
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
          </div>
        </Container>
      </section>

      <section id="experience" className="scroll-mt-32 border-y border-line/80 bg-surface/55 py-20 sm:py-24">
        <Container size="content">
          <SectionHeading
            eyebrow="Professional progression"
            title="Applied work across data science and AI/ML engineering."
            description="A concise public view of verified professional experience."
          />
          {profile.experience.length > 0 ? (
            <ol className="mt-12 grid gap-5 lg:grid-cols-2">
              {profile.experience.map((entry) => (
                <li key={`${entry.organization}-${entry.role}-${entry.period ?? "undated"}`}>
                  <Surface as="article" padding="lg" className="h-full">
                    <div className="flex items-start justify-between gap-5">
                      <Badge variant={entry.isCurrent ? "accent" : "neutral"}>
                        {entry.isCurrent ? "Current" : "Previous"}
                      </Badge>
                      {entry.period ? (
                        <span className="font-mono text-xs tracking-[0.08em] text-muted uppercase">
                          {entry.period}
                        </span>
                      ) : null}
                    </div>
                    <h3 className="mt-8 text-2xl font-semibold tracking-[-0.03em] text-ink">
                      {entry.role}
                    </h3>
                    <p className="mt-2 text-sm font-medium text-primary">
                      {entry.engagement}: {entry.organization}
                      {entry.location ? ` · ${entry.location}` : null}
                    </p>
                    <p className="mt-6 text-base leading-7 text-ink/70">
                      {entry.summary}
                    </p>
                  </Surface>
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

      <section id="education" className="scroll-mt-32 py-20 sm:py-24">
        <Container size="content">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
            <div>
              {profile.research ? (
                <>
                  <SectionHeading
                    eyebrow="Research direction"
                    title={profile.research.title}
                    description={profile.research.summary}
                    size="sm"
                  />
                  {profile.research.interests.length > 0 ? (
                    <ul className="mt-8 space-y-3">
                      {profile.research.interests.map((interest) => (
                        <li
                          key={interest}
                          className="flex gap-3 border-b border-line/70 pb-3 text-sm leading-6 text-ink/75"
                        >
                          <span
                            aria-hidden="true"
                            className="mt-2 size-1.5 shrink-0 rounded-full bg-primary"
                          />
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
            </div>

            <div>
              <SectionHeading
                eyebrow="Education"
                title="Academic foundation"
                size="sm"
              />
              {profile.education.length > 0 ? (
                <ol className="mt-8 space-y-4">
                  {profile.education.map((entry) => (
                    <li key={`${entry.institution}-${entry.degree}-${entry.period ?? "undated"}`}>
                      <Surface as="article" variant="subtle">
                        <GraduationCap
                          aria-hidden="true"
                          className="size-5 text-primary"
                        />
                        <h3 className="mt-5 text-lg font-semibold tracking-[-0.02em] text-ink">
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
          </div>
        </Container>
      </section>

      {profile.certifications.length > 0 ? (
        <section className="border-t border-line/80 py-20 sm:py-24">
          <Container size="content">
            <SectionHeading
              eyebrow="Certifications"
              title="Verified credentials"
              description="Only certifications currently published in the portfolio are shown."
            />
            <ul className="mt-12 grid gap-4 md:grid-cols-2">
              {profile.certifications.map((certification) => (
                <li key={`${certification.name}-${certification.issuer}`}>
                  <Surface as="article" className="h-full" variant="subtle">
                    <Award aria-hidden="true" className="size-5 text-primary" />
                    <h3 className="mt-5 text-lg font-semibold text-ink">
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
                        <ArrowRight aria-hidden="true" className="size-4" />
                      </a>
                    ) : null}
                  </Surface>
                </li>
              ))}
            </ul>
          </Container>
        </section>
      ) : null}

      <section id="expertise" className="scroll-mt-32 border-t border-line/80 py-20 sm:py-24">
        <Container size="content">
          <SectionHeading
            eyebrow="Technical expertise"
            title="A categorized view of tools and technical foundations."
            description="Organized for clarity, without reducing the portfolio to a wall of technology logos."
          />
          {profile.expertise.length > 0 ? (
            <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {profile.expertise.map((group) => (
                <Surface key={group.category} as="article" variant="subtle">
                  <h3 className="text-sm font-semibold tracking-[-0.01em] text-ink">
                    {group.category}
                  </h3>
                  <ul className="mt-5 flex flex-wrap gap-2" aria-label={group.category}>
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
