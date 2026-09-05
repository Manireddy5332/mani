import type { LucideIcon } from "lucide-react";
import Link from "next/link";
import {
  ArrowDownRight,
  ArrowUpRight,
  BookOpen,
  Briefcase,
  ExternalLink,
  FileText,
  FlaskConical,
  GraduationCap,
  Mail,
  MapPin,
} from "lucide-react";

import { EvidenceAtlas } from "@/components/site/system-flow";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import type { HomePageData } from "@/features/home";

type OverviewCard = {
  label: string;
  title: string;
  description: string;
  href: string;
  icon: LucideIcon;
};

function getOverviewCards(home: HomePageData): readonly OverviewCard[] {
  const currentExperience = home.experience[0];
  const education = home.education[0];

  return [
    {
      label: "Professional",
      title: currentExperience?.role ?? "Professional experience",
      description:
        currentExperience?.summary ??
        "Published professional experience will be collected here.",
      href: "/experience",
      icon: Briefcase,
    },
    {
      label: "Academic",
      title: education?.degree ?? "Academic background",
      description: education
        ? `${education.institution}${education.period ? ` · ${education.period}` : ""}`
        : "Published education records will be collected here.",
      href: "/about#education",
      icon: GraduationCap,
    },
    {
      label: "Research",
      title: home.currentResearch?.title ?? "Research interests",
      description:
        home.currentResearch?.summary ??
        "Published research interests and directions will be collected here.",
      href: "/research",
      icon: FlaskConical,
    },
    {
      label: "Selected work",
      title: home.selectedProject?.title ?? "Selected projects",
      description:
        home.selectedProject?.summary ??
        "Published project work will be collected here.",
      href: "/projects",
      icon: BookOpen,
    },
  ];
}

export type PersonalPortfolioHomeProps = Readonly<{
  home: HomePageData;
}>;

export function PersonalPortfolioHome({ home }: PersonalPortfolioHomeProps) {
  const currentExperience = home.experience[0];
  const graduateEducation = home.education[0];
  const overviewCards = getOverviewCards(home);

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <section id="about" className="relative scroll-mt-32 border-b border-line">
        <div
          aria-hidden="true"
          className="foundation-grid pointer-events-none absolute inset-0 opacity-65 [mask-image:linear-gradient(to_bottom,black,transparent_92%)]"
        />
        <Container className="relative py-20 sm:py-24 lg:py-32">
          <div className="grid items-end gap-12 lg:grid-cols-[minmax(0,1.25fr)_minmax(20rem,0.65fr)] lg:gap-16">
            <Reveal>
              <div className="max-w-5xl">
                <Badge variant="accent">Academic + professional portfolio</Badge>
                <h1 className="mt-7 text-balance font-serif text-[clamp(3.4rem,8.5vw,7.5rem)] leading-[0.88] font-medium tracking-[-0.06em] text-ink">
                  {home.identity.name}
                </h1>

                <p className="mt-8 max-w-4xl text-balance font-serif text-2xl leading-tight font-medium tracking-[-0.025em] text-primary sm:text-3xl lg:text-4xl">
                  {home.identity.positioning.join(" · ")}
                </p>
                <p className="mt-7 max-w-2xl text-pretty text-base leading-8 text-muted sm:text-lg">
                  {home.identity.introduction}
                </p>

                <div className="mt-9 flex flex-col gap-3 sm:flex-row">
                  <ButtonLink href="/experience" size="lg">
                    View professional experience
                    <ArrowDownRight aria-hidden="true" className="size-4" />
                  </ButtonLink>
                  <ButtonLink href="/research" size="lg" variant="outline">
                    Explore research interests
                  </ButtonLink>
                </div>

                <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-muted">
                  {home.identity.location ? (
                    <span className="inline-flex items-center gap-2">
                      <MapPin aria-hidden="true" className="size-4 text-secondary" />
                      {home.identity.location}
                    </span>
                  ) : null}
                  {home.links.linkedIn ? (
                    <a
                      href={home.links.linkedIn}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 font-semibold text-ink transition-colors hover:text-primary"
                    >
                      <ExternalLink aria-hidden="true" className="size-4" />
                      LinkedIn
                      <ArrowUpRight aria-hidden="true" className="size-3.5" />
                    </a>
                  ) : null}
                  {home.links.email ? (
                    <a
                      href={home.links.email}
                      className="inline-flex items-center gap-2 font-semibold text-ink transition-colors hover:text-primary"
                    >
                      <Mail aria-hidden="true" className="size-4" />
                      Email
                    </a>
                  ) : null}
                </div>
              </div>
            </Reveal>

            <Reveal delay={0.1}>
              <Surface
                as="aside"
                aria-labelledby="profile-index-title"
                className="relative overflow-hidden bg-surface/90 backdrop-blur-sm"
                padding="lg"
                variant="raised"
              >
                <span aria-hidden="true" className="absolute top-0 left-0 h-1 w-24 bg-primary" />
                <p className="eyebrow text-primary">Profile / at a glance</p>
                <h2
                  id="profile-index-title"
                  className="mt-5 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.035em]"
                >
                  Practice, study, and an emerging research direction.
                </h2>

                <dl className="mt-8 border-t border-line">
                  {currentExperience ? (
                    <div className="grid gap-2 border-b border-line py-4 sm:grid-cols-[7rem_1fr]">
                      <dt className="eyebrow text-subtle">Current role</dt>
                      <dd className="text-sm font-semibold leading-6 text-ink">
                        {currentExperience.role}
                      </dd>
                    </div>
                  ) : null}
                  {graduateEducation ? (
                    <div className="grid gap-2 border-b border-line py-4 sm:grid-cols-[7rem_1fr]">
                      <dt className="eyebrow text-subtle">Education</dt>
                      <dd className="text-sm font-semibold leading-6 text-ink">
                        {graduateEducation.degree}
                      </dd>
                    </div>
                  ) : null}
                  {home.currentResearch ? (
                    <div className="grid gap-2 border-b border-line py-4 sm:grid-cols-[7rem_1fr]">
                      <dt className="eyebrow text-subtle">Research</dt>
                      <dd className="text-sm font-semibold leading-6 text-ink">
                        {home.currentResearch.title}
                      </dd>
                    </div>
                  ) : null}
                  <div className="grid gap-2 py-4 sm:grid-cols-[7rem_1fr]">
                    <dt className="eyebrow text-subtle">Positioning</dt>
                    <dd className="text-sm font-semibold leading-6 text-ink">
                      {home.identity.positioning.join(" · ")}
                    </dd>
                  </div>
                </dl>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section className="border-b border-line py-16 sm:py-20">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Portfolio overview"
              title="A professional path shaped by study, practice, and open questions."
              description="This homepage brings the main parts of my academic and professional profile together, with dedicated pages for each area of work and study."
            />
          </Reveal>

          <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {overviewCards.map((card, index) => {
              const Icon = card.icon;

              return (
                <Reveal key={card.label} delay={index * 0.05}>
                  <Link href={card.href} className="group block h-full rounded-2xl focus-visible:outline-none">
                    <Surface
                      as="article"
                      className="h-full transition-[border-color,transform,box-shadow] duration-300 group-hover:-translate-y-1 group-hover:border-primary/45 group-hover:shadow-lift group-focus-visible:ring-2 group-focus-visible:ring-primary group-focus-visible:ring-offset-4 group-focus-visible:ring-offset-canvas motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <span className="grid size-11 place-items-center rounded-full border border-line bg-surface text-primary">
                          <Icon aria-hidden="true" className="size-5" strokeWidth={1.7} />
                        </span>
                        <span className="font-mono text-xs tracking-[0.12em] text-subtle">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                      </div>
                      <p className="eyebrow mt-10 text-secondary">{card.label}</p>
                      <h3 className="mt-3 font-serif text-2xl leading-tight font-medium tracking-[-0.03em]">
                        {card.title}
                      </h3>
                      <p className="mt-4 text-sm leading-7 text-muted">{card.description}</p>
                      <span className="mt-7 inline-flex items-center gap-2 text-sm font-semibold text-ink group-hover:text-primary">
                        Open page
                        <ArrowUpRight aria-hidden="true" className="size-4" />
                      </span>
                    </Surface>
                  </Link>
                </Reveal>
              );
            })}
          </div>
        </Container>
      </section>

      <section id="experience" className="scroll-mt-32 border-b border-line py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Professional experience"
              title={
                home.experience.length > 0
                  ? "Applied data science and AI/ML work across real operating environments."
                  : "Professional experience will appear here when published."
              }
              description={
                home.experience.length > 0
                  ? "My professional experience spans the practice areas described in the published records below."
                  : "No professional experience records are currently public."
              }
              actions={
                <ButtonLink href="/experience" variant="outline">
                  View experience page
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              }
            />
          </Reveal>

          <div className="mt-12 grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(19rem,0.8fr)] lg:gap-12">
            <Reveal>
              {home.experience.length > 0 ? (
                <ol className="border-t border-line" aria-label="Professional experience overview">
                  {home.experience.map((item, index) => (
                  <li
                    key={`${item.organization}-${item.role}`}
                    className="grid gap-5 border-b border-line py-8 sm:grid-cols-[8rem_minmax(0,1fr)] sm:py-10"
                  >
                    <div>
                      {item.period ? (
                        <span className="font-mono text-xs font-semibold tracking-[0.1em] text-primary uppercase">
                          {item.period}
                        </span>
                      ) : null}
                      <p className="mt-2 text-xs text-subtle">
                        Entry {String(index + 1).padStart(2, "0")}
                      </p>
                    </div>
                    <article>
                      <h3 className="font-serif text-2xl font-medium tracking-[-0.03em] sm:text-3xl">
                        {item.role}
                      </h3>
                      <p className="mt-2 font-semibold text-ink">
                        {item.engagement}: {item.organization}
                        {item.location ? (
                          <span className="font-normal text-muted"> · {item.location}</span>
                        ) : null}
                      </p>
                      <p className="mt-5 max-w-2xl text-base leading-8 text-muted">
                        {item.summary}
                      </p>
                    </article>
                  </li>
                  ))}
                </ol>
              ) : (
                <Surface padding="lg" variant="subtle">
                  <p className="text-sm leading-7 text-muted">
                    No professional experience records are currently public.
                  </p>
                </Surface>
              )}
            </Reveal>

            <Reveal delay={0.08}>
              <Surface
                id="education"
                as="aside"
                aria-labelledby="education-title"
                className="scroll-mt-32 lg:sticky lg:top-32 lg:self-start"
                padding="lg"
                variant="accent"
              >
                <div className="flex items-center gap-3 text-primary">
                  <GraduationCap aria-hidden="true" className="size-5" />
                  <p className="eyebrow">Education</p>
                </div>
                <h2
                  id="education-title"
                  className="mt-5 font-serif text-3xl leading-tight font-medium tracking-[-0.035em]"
                >
                  Academic background
                </h2>
                <div className="mt-7 divide-y divide-line border-y border-line">
                  {home.education.length > 0 ? home.education.map((item) => (
                    <article key={item.degree} className="py-6">
                      {item.period ? (
                        <p className="font-mono text-xs font-semibold tracking-[0.1em] text-secondary uppercase">
                          {item.period}
                        </p>
                      ) : null}
                      <h3 className="mt-3 text-base font-semibold leading-6 text-ink">
                        {item.degree}
                      </h3>
                      <p className="mt-2 text-sm leading-6 text-muted">{item.institution}</p>
                    </article>
                  )) : (
                    <p className="py-6 text-sm leading-6 text-muted">
                      No education records are currently public.
                    </p>
                  )}
                </div>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section id="research" className="scroll-mt-32 border-b border-line py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Research interests"
              title="Questions from professional practice, developed through academic inquiry."
              description="My current interests center on how generative AI is adopted, compared, evaluated, and made dependable outside controlled demonstrations."
              actions={
                <ButtonLink href="/research" variant="outline">
                  View research page
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              }
            />
          </Reveal>

          <div className="mt-12 grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(19rem,0.85fr)]">
            <Reveal>
              {home.currentResearch ? (
                <Surface
                  id="writing"
                  as="article"
                  className="scroll-mt-32"
                  padding="lg"
                  variant="raised"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <Badge variant="accent">Research in progress</Badge>
                    <span className="eyebrow text-muted">{home.currentResearch.stage}</span>
                  </div>
                  <h3 className="mt-6 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.04em] sm:text-4xl">
                    {home.currentResearch.title}
                  </h3>
                  {home.currentResearch.format ? (
                    <p className="mt-4 font-mono text-xs font-semibold tracking-[0.08em] text-secondary uppercase">
                      {home.currentResearch.format}
                    </p>
                  ) : null}
                  <p className="mt-6 max-w-3xl text-base leading-8 text-muted">
                    {home.currentResearch.summary}
                  </p>

                  {home.currentResearch.questions.length > 0 ? (
                    <ol className="mt-8 grid gap-3" aria-label="Current research questions">
                      {home.currentResearch.questions.map((question, index) => (
                        <li
                          key={question}
                          className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 border-t border-line pt-4 text-sm leading-6 text-ink"
                        >
                          <span className="font-mono text-xs text-primary">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <span>{question}</span>
                        </li>
                      ))}
                    </ol>
                  ) : null}
                </Surface>
              ) : (
                <Surface id="writing" className="scroll-mt-32" padding="lg" variant="subtle">
                  <p className="eyebrow text-primary">Research direction</p>
                  <h3 className="mt-5 font-serif text-3xl font-medium tracking-[-0.035em]">
                    No current research direction is published.
                  </h3>
                </Surface>
              )}
            </Reveal>

            <Reveal delay={0.08}>
              <Surface as="aside" aria-labelledby="interest-list-title" className="h-full" padding="lg" variant="subtle">
                <p className="eyebrow text-primary">Areas of interest</p>
                <h3
                  id="interest-list-title"
                  className="mt-5 font-serif text-3xl font-medium tracking-[-0.035em]"
                >
                  Current research themes
                </h3>
                {home.researchInterests.length > 0 ? (
                  <ol className="mt-7 border-t border-line">
                    {home.researchInterests.map((interest, index) => (
                    <li
                      key={interest}
                      className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 border-b border-line py-5"
                    >
                      <span className="font-mono text-xs text-subtle">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <span className="text-sm font-semibold leading-6 text-ink">{interest}</span>
                    </li>
                    ))}
                  </ol>
                ) : (
                  <p className="mt-7 border-t border-line pt-5 text-sm leading-6 text-muted">
                    No research interests are currently public.
                  </p>
                )}
              </Surface>
            </Reveal>
          </div>

          <Reveal className="mt-10" delay={0.08}>
            <EvidenceAtlas
              headingLevel="h3"
              title="A research map from question to evidence."
              description="Evidence Atlas is a framework for tracing each line of inquiry through its methods, technologies, related projects, and supported findings as verified work develops."
            />
          </Reveal>
        </Container>
      </section>

      <section id="projects" className="scroll-mt-32 border-b border-line py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Selected project + technical toolkit"
              title="Academic work supported by a broad applied-technology foundation."
              description="The project below is drawn directly from my academic CV. The technical toolkit reflects the skills documented across my education and professional experience."
              actions={
                <ButtonLink href="/projects" variant="outline">
                  View projects page
                  <ArrowUpRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              }
            />
          </Reveal>

          <div className="mt-12 grid gap-6 lg:grid-cols-[minmax(0,1.12fr)_minmax(20rem,0.88fr)] lg:items-start">
            <Reveal>
              {home.selectedProject ? (
                <Surface as="article" padding="lg" variant="raised">
                  <div className="flex flex-wrap items-center gap-3">
                    {home.selectedProject.type ? (
                      <Badge variant="accent">{home.selectedProject.type}</Badge>
                    ) : null}
                    <span className="eyebrow text-muted">Selected academic project</span>
                  </div>
                  <h3 className="mt-6 text-balance font-serif text-3xl leading-tight font-medium tracking-[-0.04em] sm:text-4xl">
                    {home.selectedProject.title}
                  </h3>
                  {home.selectedProject.institution ? (
                    <p className="mt-5 text-sm font-semibold leading-6 text-ink">
                      {home.selectedProject.institution}
                      {home.selectedProject.advisor ? (
                        <span className="font-normal text-muted">
                          {" "}· Advisor: {home.selectedProject.advisor}
                        </span>
                      ) : null}
                    </p>
                  ) : null}
                  {home.selectedProject.role ? (
                    <p className="mt-3 font-mono text-xs font-semibold tracking-[0.08em] text-secondary uppercase">
                      {home.selectedProject.role}
                    </p>
                  ) : null}
                  <p className="mt-6 max-w-3xl text-base leading-8 text-muted">
                    {home.selectedProject.summary}
                  </p>
                  {home.selectedProject.technologies.length > 0 ? (
                    <ul className="mt-7 flex flex-wrap gap-2" aria-label="Project technologies">
                      {home.selectedProject.technologies.map((technology) => (
                        <li key={technology}>
                          <Badge variant="outline">{technology}</Badge>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  <div className="mt-8 flex flex-wrap gap-3">
                    <ButtonLink href={home.selectedProject.href} variant="outline">
                      View case study
                      <ArrowUpRight aria-hidden="true" className="size-4" />
                    </ButtonLink>
                    {home.selectedProject.repositoryUrl ? (
                      <ButtonLink
                        href={home.selectedProject.repositoryUrl}
                        target="_blank"
                        rel="noreferrer"
                        variant="ghost"
                      >
                        <ExternalLink aria-hidden="true" className="size-4" />
                        Project repository
                        <ArrowUpRight aria-hidden="true" className="size-4" />
                      </ButtonLink>
                    ) : null}
                  </div>
                </Surface>
              ) : (
                <Surface padding="lg" variant="subtle">
                  <p className="eyebrow text-primary">Selected work</p>
                  <h3 className="mt-5 font-serif text-3xl font-medium tracking-[-0.035em]">
                    No project is currently published.
                  </h3>
                </Surface>
              )}
            </Reveal>

            <Reveal delay={0.08}>
              <Surface as="aside" aria-labelledby="technical-toolkit-title" padding="lg" variant="subtle">
                <p className="eyebrow text-primary">Technical toolkit</p>
                <h3
                  id="technical-toolkit-title"
                  className="mt-5 font-serif text-3xl font-medium tracking-[-0.035em]"
                >
                  Skills documented in the academic CV
                </h3>
                <div className="mt-7 divide-y divide-line border-y border-line">
                  {home.technicalExpertise.length > 0 ? home.technicalExpertise.map((group) => (
                    <section key={group.category} className="py-5">
                      <h4 className="text-sm font-semibold text-ink">{group.category}</h4>
                      <p className="mt-2 text-sm leading-6 text-muted">
                        {group.skills.join(" · ")}
                      </p>
                    </section>
                  )) : (
                    <p className="py-5 text-sm leading-6 text-muted">
                      No technical skill groups are currently public.
                    </p>
                  )}
                </div>
              </Surface>
            </Reveal>
          </div>
        </Container>
      </section>

      <section id="contact" className="scroll-mt-32 py-20 sm:py-24 lg:py-32">
        <Container>
          <Reveal>
            <Surface className="relative overflow-hidden" padding="lg" variant="accent">
              <div
                aria-hidden="true"
                className="foundation-grid pointer-events-none absolute inset-0 opacity-45 [mask-image:linear-gradient(to_right,black,transparent_80%)]"
              />
              <div className="relative grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_auto] lg:items-end">
                <div>
                  <div className="flex items-center gap-3 text-primary">
                    <FileText aria-hidden="true" className="size-5" />
                    <p className="eyebrow">Academic CV + contact</p>
                  </div>
                  <h2 className="mt-5 max-w-4xl text-balance font-serif text-4xl leading-[1.02] font-medium tracking-[-0.045em] sm:text-5xl lg:text-6xl">
                    Continue the conversation around applied AI, research, and doctoral study.
                  </h2>
                  <p className="mt-6 max-w-2xl text-base leading-8 text-muted">
                    This homepage provides a concise overview. For the complete
                    chronology of my education, professional work, skills, and
                    academic project, request my academic CV or connect with me
                    professionally.
                  </p>
                  {home.identity.location ? (
                    <p className="mt-5 inline-flex items-center gap-2 text-sm text-muted">
                      <MapPin aria-hidden="true" className="size-4 text-secondary" />
                      {home.identity.location}
                    </p>
                  ) : null}
                </div>

                <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
                  <ButtonLink href="/resume" size="lg">
                    <FileText aria-hidden="true" className="size-4" />
                    View academic CV
                    <ArrowUpRight aria-hidden="true" className="size-4" />
                  </ButtonLink>
                  {home.links.linkedIn ? (
                    <ButtonLink
                      href={home.links.linkedIn}
                      target="_blank"
                      rel="noreferrer"
                      size="lg"
                      variant="outline"
                    >
                      <ExternalLink aria-hidden="true" className="size-4" />
                      LinkedIn profile
                      <ArrowUpRight aria-hidden="true" className="size-4" />
                    </ButtonLink>
                  ) : null}
                  <ButtonLink href="/contact" size="lg" variant="ghost">
                    <Mail aria-hidden="true" className="size-4" />
                    Contact
                    <ArrowUpRight aria-hidden="true" className="size-4" />
                  </ButtonLink>
                </div>
              </div>
            </Surface>
          </Reveal>
        </Container>
      </section>
    </main>
  );
}
