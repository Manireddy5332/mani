import type { LucideIcon } from "lucide-react";
import Link from "next/link";
import {
  ArrowDownRight,
  ArrowUpRight,
  BookOpen,
  Briefcase,
  CircleDotDashed,
  ExternalLink,
  FileText,
  FlaskConical,
  GraduationCap,
  Mail,
  MapPin,
  Sparkles,
} from "lucide-react";

import { ResearchConstellation } from "@/components/site/research-constellation";
import { ProfilePhoto } from "@/components/site/profile-photo";
import { EvidenceAtlas } from "@/components/site/system-flow";
import {
  Badge,
  ButtonLink,
  Container,
  Reveal,
  SectionHeading,
  Surface,
} from "@/components/ui";
import { ACADEMIC_CV_LINK } from "@/lib/academic-cv";
import { cn } from "@/lib/cn";
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
  const publicSignals = [
    { label: "Experience", value: home.experience.length },
    { label: "Research themes", value: home.researchInterests.length },
    { label: "Skill groups", value: home.technicalExpertise.length },
  ] as const;

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <section
        id="about"
        className="relative isolate scroll-mt-32 overflow-hidden border-b border-line bg-canvas/40"
      >
        <div
          aria-hidden="true"
          className="foundation-grid pointer-events-none absolute inset-0 opacity-75 [mask-image:radial-gradient(ellipse_88%_88%_at_52%_12%,black,transparent)]"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-56 -right-48 size-[42rem] rounded-full border border-primary/15 bg-primary/[0.035]"
        />
        <Container className="relative py-20 sm:py-28 lg:min-h-[calc(100svh-5rem)] lg:py-32">
          <div
            className={cn(
              "grid min-h-full items-center gap-14 lg:grid-cols-[minmax(0,1.25fr)_minmax(21rem,0.68fr)] lg:gap-20",
              home.identity.photo && "items-start",
            )}
          >
            <Reveal>
              <div className="max-w-5xl">
                <div className="flex flex-wrap items-center gap-3">
                  <Badge variant="accent">Academic + professional portfolio</Badge>
                  <span className="hidden h-px w-16 bg-gradient-to-r from-primary to-transparent sm:block" />
                  <span className="eyebrow text-subtle">AI / ML</span>
                </div>
                <h1 className="mt-8 max-w-5xl text-balance font-serif text-[clamp(3.75rem,8.8vw,8.25rem)] leading-[0.84] font-medium tracking-[-0.065em] text-ink">
                  {home.identity.name}
                </h1>

                <p className="mt-9 flex max-w-4xl flex-wrap items-center gap-x-3 gap-y-2 text-balance font-serif text-2xl leading-tight font-medium tracking-[-0.025em] text-primary sm:text-3xl lg:text-4xl">
                  {home.identity.positioning.map((position, index) => (
                    <span key={`${position}-${index}`} className="contents">
                      {index > 0 ? (
                        <span aria-hidden="true" className="text-secondary/70">
                          /
                        </span>
                      ) : null}
                      <span>{position}</span>
                    </span>
                  ))}
                </p>
                <p className="mt-8 max-w-2xl border-l-2 border-primary/55 pl-5 text-pretty text-base leading-8 text-muted sm:pl-7 sm:text-lg">
                  {home.identity.introduction}
                </p>

                <div className="mt-10 flex flex-col gap-3 sm:flex-row">
                  <ButtonLink href="/experience" size="lg">
                    View professional experience
                    <ArrowDownRight aria-hidden="true" className="size-4" />
                  </ButtonLink>
                  <ButtonLink href="/research" size="lg" variant="outline">
                    Explore research interests
                  </ButtonLink>
                </div>

                <div className="mt-9 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-muted">
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
                      className="group/link inline-flex min-h-11 items-center gap-2 font-semibold text-ink transition-colors hover:text-primary motion-reduce:transition-none"
                    >
                      <ExternalLink aria-hidden="true" className="size-4" />
                      LinkedIn
                      <ArrowUpRight
                        aria-hidden="true"
                        className="size-3.5 transition-transform group-hover/link:-translate-y-0.5 group-hover/link:translate-x-0.5 motion-reduce:transform-none motion-reduce:transition-none"
                      />
                    </a>
                  ) : null}
                  {home.links.email ? (
                    <a
                      href={home.links.email}
                      className="inline-flex min-h-11 items-center gap-2 font-semibold text-ink transition-colors hover:text-primary motion-reduce:transition-none"
                    >
                      <Mail aria-hidden="true" className="size-4" />
                      Email
                    </a>
                  ) : null}
                </div>
              </div>
            </Reveal>

            {home.identity.photo ? (
              <Reveal className="relative lg:pt-12" delay={0.1}>
                <ResearchConstellation className="pointer-events-none absolute -top-28 -right-36 hidden h-[38rem] w-[42rem] max-w-none opacity-90 lg:block" />
                <ProfilePhoto photo={home.identity.photo} variant="hero" />
              </Reveal>
            ) : null}

            <Reveal
              className={cn("relative", home.identity.photo && "lg:col-span-2")}
              delay={0.1}
            >
              {!home.identity.photo ? (
                <ResearchConstellation className="pointer-events-none absolute -top-28 -right-36 hidden h-[38rem] w-[42rem] max-w-none opacity-90 lg:block" />
              ) : null}
              <div className={cn("relative", !home.identity.photo && "lg:py-10")}>
                <Surface
                  as="aside"
                  aria-labelledby="profile-index-title"
                  className="relative overflow-hidden border-primary/20 bg-surface/82 shadow-[0_38px_100px_-55px_rgb(var(--ds-shadow-color)/0.75)] backdrop-blur-xl"
                  padding="lg"
                  variant="raised"
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-secondary to-transparent"
                  />
                  <div className="flex items-center justify-between gap-4">
                    <p className="eyebrow text-primary">Profile / at a glance</p>
                    <CircleDotDashed aria-hidden="true" className="size-5 text-secondary" />
                  </div>
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

                  <div className="mt-7 grid grid-cols-3 gap-2 border-t border-line pt-6">
                    {publicSignals.map((signal) => (
                      <div key={signal.label} className="min-w-0">
                        <p className="font-serif text-3xl leading-none font-medium tabular-nums text-primary">
                          {String(signal.value).padStart(2, "0")}
                        </p>
                        <p className="mt-2 text-[0.67rem] leading-4 font-semibold tracking-[0.08em] text-subtle uppercase">
                          {signal.label}
                        </p>
                      </div>
                    ))}
                  </div>
                </Surface>
              </div>
            </Reveal>
          </div>

          <div
            aria-hidden="true"
            className="mt-14 flex items-center gap-4 text-primary sm:mt-18"
          >
            <span className="size-2 rounded-full bg-primary" />
            <span className="h-px w-20 bg-gradient-to-r from-primary to-transparent" />
            <Sparkles className="size-4" strokeWidth={1.6} />
          </div>
        </Container>
      </section>

      <section className="relative border-b border-line bg-surface/30 py-18 sm:py-22">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/35 to-transparent"
        />
        <Container>
          <Reveal>
            <SectionHeading
              eyebrow="Portfolio overview"
              title="A professional path shaped by study, practice, and open questions."
              description="This homepage brings the main parts of my academic and professional profile together, with dedicated pages for each area of work and study."
            />
          </Reveal>

          <div className="mt-12 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {overviewCards.map((card, index) => {
              const Icon = card.icon;

              return (
                <Reveal key={card.label} delay={index * 0.05}>
                  <Link
                    href={card.href}
                    className="group block h-full rounded-3xl focus-visible:outline-none"
                  >
                    <Surface
                      as="article"
                      className="relative h-full min-h-80 overflow-hidden rounded-3xl bg-canvas/75 transition-[border-color,transform,box-shadow,background-color] duration-300 group-hover:-translate-y-1.5 group-hover:border-primary/45 group-hover:bg-surface group-hover:shadow-lift group-focus-visible:ring-2 group-focus-visible:ring-primary group-focus-visible:ring-offset-4 group-focus-visible:ring-offset-canvas motion-reduce:transform-none motion-reduce:transition-none"
                      padding="lg"
                      variant="subtle"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute inset-x-0 top-0 h-0.5 origin-left scale-x-0 bg-gradient-to-r from-primary via-secondary to-transparent transition-transform duration-300 group-hover:scale-x-100 motion-reduce:transition-none"
                      />
                      <div className="flex items-center justify-between gap-4">
                        <span className="grid size-12 place-items-center rounded-2xl border border-line bg-surface text-primary shadow-sm transition-[border-color,transform] duration-300 group-hover:rotate-2 group-hover:scale-105 group-hover:border-primary/55 motion-reduce:transform-none motion-reduce:transition-none">
                          <Icon aria-hidden="true" className="size-5" strokeWidth={1.7} />
                        </span>
                        <span className="font-mono text-xs tracking-[0.12em] text-subtle">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                      </div>
                      <p className="eyebrow mt-12 text-secondary">{card.label}</p>
                      <h3 className="mt-3 font-serif text-2xl leading-tight font-medium tracking-[-0.035em] sm:text-[1.7rem]">
                        {card.title}
                      </h3>
                      <p className="mt-4 text-sm leading-7 text-muted">{card.description}</p>
                      <span className="mt-8 inline-flex items-center gap-2 text-sm font-semibold text-ink transition-colors group-hover:text-primary motion-reduce:transition-none">
                        Open page
                        <ArrowUpRight
                          aria-hidden="true"
                          className="size-4 transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 motion-reduce:transform-none motion-reduce:transition-none"
                        />
                      </span>
                    </Surface>
                  </Link>
                </Reveal>
              );
            })}
          </div>
        </Container>
      </section>

      <section
        id="experience"
        className="relative scroll-mt-32 border-b border-line py-20 sm:py-24 lg:py-32"
      >
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
                <ol
                  className="relative before:absolute before:top-5 before:bottom-8 before:left-[0.42rem] before:w-px before:bg-gradient-to-b before:from-primary before:via-line-strong before:to-transparent sm:before:left-[8.42rem]"
                  aria-label="Professional experience overview"
                >
                  {home.experience.map((item, index) => (
                    <li
                      key={`${item.organization}-${item.role}`}
                      className="relative grid gap-5 border-b border-line/80 py-8 pl-8 sm:grid-cols-[8rem_minmax(0,1fr)] sm:py-10 sm:pl-0"
                    >
                      <span
                        aria-hidden="true"
                        className="absolute top-10 left-0 size-3.5 rounded-full border-[3px] border-canvas bg-primary shadow-[0_0_0_1px_var(--ds-primary)] sm:left-[8rem]"
                      />
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
                      <article className="sm:pl-8">
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
                className="relative scroll-mt-32 overflow-hidden border-primary/25 lg:sticky lg:top-32 lg:self-start"
                padding="lg"
                variant="accent"
              >
                <span
                  aria-hidden="true"
                  className="absolute top-0 right-8 left-8 h-px bg-gradient-to-r from-transparent via-primary to-transparent"
                />
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

      <section
        id="research"
        className="relative scroll-mt-32 border-b border-line bg-ink/[0.018] py-20 sm:py-24 lg:py-32"
      >
        <div
          aria-hidden="true"
          className="pointer-events-none absolute top-0 right-0 h-80 w-80 rounded-full bg-secondary/[0.07] blur-3xl"
        />
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
                  className="group relative scroll-mt-32 overflow-hidden border-primary/20 bg-surface/90"
                  padding="lg"
                  variant="raised"
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-secondary to-transparent"
                  />
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
                          className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-3 rounded-xl border border-line/70 bg-canvas/55 p-4 text-sm leading-6 text-ink transition-[border-color,transform] duration-300 hover:translate-x-1 hover:border-primary/35 motion-reduce:transform-none motion-reduce:transition-none"
                        >
                          <span className="grid size-8 place-items-center rounded-full bg-primary/[0.08] font-mono text-xs text-primary">
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
              <Surface
                as="aside"
                aria-labelledby="interest-list-title"
                className="relative h-full overflow-hidden bg-canvas/70"
                padding="lg"
                variant="subtle"
              >
                <span
                  aria-hidden="true"
                  className="absolute -top-20 -right-20 size-44 rounded-full border border-secondary/20 bg-secondary/[0.04]"
                />
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
                        className="group/interest grid grid-cols-[2rem_minmax(0,1fr)] gap-3 border-b border-line py-5 transition-colors hover:border-primary/40 motion-reduce:transition-none"
                      >
                        <span className="font-mono text-xs text-subtle transition-colors group-hover/interest:text-primary motion-reduce:transition-none">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="text-sm font-semibold leading-6 text-ink">
                          {interest}
                        </span>
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

      <section
        id="projects"
        className="relative scroll-mt-32 border-b border-line py-20 sm:py-24 lg:py-32"
      >
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
                <Surface
                  as="article"
                  className="group relative overflow-hidden border-primary/20 bg-surface/90"
                  padding="lg"
                  variant="raised"
                >
                  <div
                    aria-hidden="true"
                    className="pointer-events-none absolute -top-32 -right-28 size-72 rounded-full bg-primary/[0.09] blur-3xl transition-transform duration-700 group-hover:scale-110 motion-reduce:transform-none motion-reduce:transition-none"
                  />
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-secondary to-transparent"
                  />
                  <div className="relative">
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
                      <ul
                        className="mt-7 flex flex-wrap gap-2"
                        aria-label="Project technologies"
                      >
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
              <Surface
                as="aside"
                aria-labelledby="technical-toolkit-title"
                className="relative overflow-hidden bg-ink/[0.025]"
                padding="lg"
                variant="subtle"
              >
                <p className="eyebrow text-primary">Technical toolkit</p>
                <h3
                  id="technical-toolkit-title"
                  className="mt-5 font-serif text-3xl font-medium tracking-[-0.035em]"
                >
                  Skills documented in the academic CV
                </h3>
                <div className="mt-7 divide-y divide-line border-y border-line">
                  {home.technicalExpertise.length > 0 ? (
                    home.technicalExpertise.map((group) => (
                      <section
                        key={group.category}
                        className="group/skill py-5 transition-[padding] duration-300 hover:pl-2 motion-reduce:transition-none"
                      >
                        <h4 className="text-sm font-semibold text-ink">{group.category}</h4>
                        <p className="mt-2 text-sm leading-6 text-muted">
                          {group.skills.join(" · ")}
                        </p>
                      </section>
                    ))
                  ) : (
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

      <section
        id="contact"
        className="relative scroll-mt-32 bg-surface/25 py-20 sm:py-24 lg:py-32"
      >
        <Container>
          <Reveal>
            <Surface
              className="relative overflow-hidden rounded-3xl border-primary/30 bg-primary/[0.065] shadow-[0_32px_90px_-52px_rgb(var(--ds-shadow-color)/0.6)]"
              padding="lg"
              variant="accent"
            >
              <div
                aria-hidden="true"
                className="foundation-grid pointer-events-none absolute inset-0 opacity-45 [mask-image:linear-gradient(to_right,black,transparent_80%)]"
              />
              <div
                aria-hidden="true"
                className="pointer-events-none absolute -right-24 -bottom-36 size-80 rounded-full border border-secondary/25 bg-secondary/[0.06]"
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
                    academic project, view my academic CV, or connect with me
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
                  <ButtonLink {...ACADEMIC_CV_LINK} size="lg">
                    <FileText aria-hidden="true" className="size-4" />
                    View Academic CV
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
