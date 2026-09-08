import {
  ArrowUpRight,
  Award,
  BriefcaseBusiness,
  Download,
  FileText,
  GraduationCap,
  Mail,
  Microscope,
} from "lucide-react";

import {
  Badge,
  ButtonLink,
  Container,
  SectionHeading,
  Surface,
} from "@/components/ui";

import type { ResumePageData } from "../types";

export type ResumePageProps = {
  readonly resume: ResumePageData;
};

export function ResumePage({ resume }: ResumePageProps) {
  return (
    <main id="main-content">
      <section className="foundation-grid border-b border-line/80 py-20 sm:py-24 lg:py-28">
        <Container size="content">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-end">
            <div>
              <Badge variant="accent">Academic + professional CV</Badge>
              <h1 className="mt-7 text-balance font-serif text-5xl leading-[0.98] font-medium tracking-[-0.045em] text-ink sm:text-6xl lg:text-7xl">
                {resume.name}
              </h1>
              <p className="mt-6 max-w-3xl text-pretty text-lg leading-8 text-ink/75 sm:text-xl">
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

            <Surface as="aside" variant="accent" padding="lg">
              <FileText aria-hidden="true" className="size-6 text-primary" />
              <h2 className="mt-5 text-xl font-semibold tracking-[-0.02em] text-ink">
                {resume.cvDownloadUrl
                  ? "Current academic CV"
                  : "Academic CV available by request"}
              </h2>
              <p className="mt-3 text-sm leading-6 text-muted">
                {resume.cvDownloadUrl
                  ? "Open the current published document for academic or professional review."
                  : "Email me to receive the current document for academic or professional review."}
              </p>
              {resume.cvDownloadUrl ? (
                <ButtonLink
                  className="mt-6"
                  href={resume.cvDownloadUrl}
                  rel="noreferrer"
                  target="_blank"
                  wide
                >
                  <Download aria-hidden="true" className="size-4" />
                  Open current CV
                </ButtonLink>
              ) : resume.cvRequestUrl ? (
                <ButtonLink className="mt-6" href={resume.cvRequestUrl} wide>
                  <Mail aria-hidden="true" className="size-4" />
                  Request by email
                </ButtonLink>
              ) : (
                <ButtonLink className="mt-6" href="/contact" wide>
                  Contact me
                </ButtonLink>
              )}
            </Surface>
          </div>
        </Container>
      </section>

      {resume.location || resume.emailAddress || resume.linkedInUrl ? (
        <section className="py-20 sm:py-24">
          <Container size="content">
            <div className="grid gap-6 md:grid-cols-3">
            {resume.location ? (
              <Surface variant="subtle">
                <p className="eyebrow text-primary">Location</p>
                <p className="mt-3 text-sm leading-6 text-ink/75">
                  {resume.location}
                </p>
              </Surface>
            ) : null}
            {resume.emailAddress ? (
              <Surface variant="subtle">
                <p className="eyebrow text-primary">Email</p>
                <a
                  className="mt-3 block break-all text-sm leading-6 text-ink/75 underline decoration-line underline-offset-4 transition-colors hover:text-primary"
                  href={`mailto:${resume.emailAddress}`}
                >
                  {resume.emailAddress}
                </a>
              </Surface>
            ) : null}
            {resume.linkedInUrl ? (
              <Surface variant="subtle">
                <p className="eyebrow text-primary">Professional profile</p>
                <a
                  className="mt-3 inline-flex items-center gap-2 text-sm leading-6 text-ink/75 underline decoration-line underline-offset-4 transition-colors hover:text-primary"
                  href={resume.linkedInUrl}
                  rel="noreferrer"
                  target="_blank"
                >
                  LinkedIn
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </a>
              </Surface>
            ) : null}
            </div>
          </Container>
        </section>
      ) : null}

      <section className="border-y border-line/80 bg-surface/55 py-20 sm:py-24">
        <Container size="content">
          <SectionHeading
            eyebrow="Professional experience"
            title="Data science and AI/ML engineering"
            description="Concise professional summaries prepared for academic and technical review."
          />
          {resume.experience.length > 0 ? (
            <ol className="mt-12 space-y-5">
              {resume.experience.map((entry) => (
                <li key={`${entry.organization}-${entry.role}-${entry.period ?? "undated"}`}>
                  <Surface as="article" padding="lg">
                    <div className="grid gap-6 md:grid-cols-[12rem_minmax(0,1fr)]">
                      <div>
                        <BriefcaseBusiness
                          aria-hidden="true"
                          className="size-5 text-primary"
                        />
                        {entry.period ? (
                          <p className="mt-4 font-mono text-xs tracking-[0.08em] text-muted uppercase">
                            {entry.period}
                          </p>
                        ) : null}
                        <Badge className="mt-4" variant={entry.isCurrent ? "accent" : "neutral"}>
                          {entry.isCurrent ? "Current" : "Previous"}
                        </Badge>
                      </div>
                      <div>
                        <h3 className="text-2xl font-semibold tracking-[-0.03em] text-ink">
                          {entry.role}
                        </h3>
                        <p className="mt-2 text-sm font-medium text-primary">
                          {entry.engagement}: {entry.organization}
                          {entry.location ? ` · ${entry.location}` : null}
                        </p>
                        <p className="mt-5 text-base leading-7 text-ink/70">
                          {entry.summary}
                        </p>
                      </div>
                    </div>
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

      <section className="py-20 sm:py-24">
        <Container size="content">
          <div className="grid gap-14 lg:grid-cols-2 lg:gap-16">
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

            <div>
              {resume.research ? (
                <>
                  <SectionHeading
                    eyebrow="Research direction"
                    title={resume.research.format ?? resume.research.title}
                    description={resume.research.summary}
                    size="sm"
                  />
                  <Surface className="mt-8" variant="accent">
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
        <section className="border-y border-line/80 bg-surface/55 py-20 sm:py-24">
          <Container size="content">
            <SectionHeading
              eyebrow="Selected academic project"
              title={resume.project.title}
              description={resume.project.summary}
            />
            <Surface className="mt-10" padding="lg">
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
        <section className="py-20 sm:py-24">
          <Container size="content">
            <SectionHeading
              eyebrow="Certifications"
              title="Verified credentials"
              description="Only certifications currently published in the portfolio are shown."
            />
            <ul className="mt-12 grid gap-4 md:grid-cols-2">
              {resume.certifications.map((certification) => (
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

      <section className="py-20 sm:py-24">
        <Container size="content">
          <SectionHeading
            eyebrow="Technical expertise"
            title="Skills and platforms"
            description="The categorized technical areas represented in my academic CV."
          />
          {resume.expertise.length > 0 ? (
            <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {resume.expertise.map((group) => (
                <Surface key={group.category} as="article" variant="subtle">
                  <h3 className="text-sm font-semibold text-ink">{group.category}</h3>
                  <p className="mt-4 text-sm leading-7 text-muted">
                    {group.skills.join(" · ")}
                  </p>
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
