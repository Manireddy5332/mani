import {
  ArrowRight,
  ArrowUpRight,
  Code2,
  ExternalLink,
  Globe2,
  Link2,
  Mail,
  MapPin,
  UserRound,
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

import type { ContactMethod, ContactPageData } from "../types";

const methodIcons = {
  email: Mail,
  linkedin: UserRound,
  github: Code2,
  website: Globe2,
  other: Link2,
  location: MapPin,
  repository: ExternalLink,
} as const;

function ContactMethodCard({
  index,
  method,
}: {
  readonly index: number;
  readonly method: ContactMethod;
}) {
  const Icon = methodIcons[method.kind];

  return (
    <Surface
      as="article"
      padding="lg"
      className={`group relative isolate h-full overflow-hidden transition-[border-color,box-shadow,transform] duration-300 ease-out focus-within:border-primary/45 focus-within:ring-2 focus-within:ring-primary/30 focus-within:ring-offset-4 focus-within:ring-offset-canvas motion-reduce:transform-none motion-reduce:transition-none ${
        method.href
          ? "hover:-translate-y-1 hover:border-primary/35 hover:shadow-lift"
          : "hover:border-line-strong"
      }`}
    >
      <span
        aria-hidden="true"
        className="absolute inset-x-0 top-0 h-px origin-left scale-x-0 bg-gradient-to-r from-primary via-secondary to-transparent transition-transform duration-300 group-hover:scale-x-100 group-focus-within:scale-x-100 motion-reduce:transition-none"
      />
      <div className="flex items-start justify-between gap-4">
        <div className="grid size-11 place-items-center rounded-xl border border-primary/20 bg-primary/10 text-primary transition-[background-color,border-color,transform] duration-300 group-hover:-translate-y-0.5 group-hover:border-primary/35 group-hover:bg-primary/15 group-focus-within:border-primary/35 motion-reduce:transform-none motion-reduce:transition-none">
          <Icon aria-hidden="true" className="size-5" />
        </div>
        <span className="font-mono text-[0.68rem] font-semibold tracking-[0.14em] text-subtle">
          {String(index + 1).padStart(2, "0")}
        </span>
      </div>
      <h3 className="mt-6 eyebrow text-muted">{method.label}</h3>
      {method.href ? (
        <a
          className="mt-3 inline-flex max-w-full items-center gap-2 break-all rounded-sm text-lg font-semibold tracking-[-0.02em] text-ink underline decoration-line underline-offset-4 transition-colors after:absolute after:inset-0 after:z-10 after:rounded-2xl hover:text-primary focus-visible:outline-none motion-reduce:transition-none"
          href={method.href}
          aria-label={
            method.external
              ? `${method.label}: ${method.value} (opens in a new tab)`
              : undefined
          }
          rel={method.external ? "noreferrer" : undefined}
          target={method.external ? "_blank" : undefined}
        >
          {method.value}
          {method.external ? (
            <ArrowUpRight
              aria-hidden="true"
              className="size-4 shrink-0 transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 motion-reduce:transform-none motion-reduce:transition-none"
            />
          ) : null}
        </a>
      ) : (
        <p className="mt-3 text-lg font-semibold tracking-[-0.02em] text-ink">
          {method.value}
        </p>
      )}
      <p className="mt-4 text-sm leading-6 text-muted">{method.description}</p>
    </Surface>
  );
}

export type ContactPageProps = {
  readonly contact: ContactPageData;
};

export function ContactPage({ contact }: ContactPageProps) {
  const emailMethod = contact.methods.find((method) => method.kind === "email");

  return (
    <main id="main-content" className="flex-1 overflow-hidden">
      <section className="relative isolate overflow-hidden border-b border-line/80 py-20 sm:py-24 lg:py-28">
        <div
          aria-hidden="true"
          className="foundation-grid pointer-events-none absolute inset-0 -z-20 opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent_94%)]"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-36 right-[-8rem] -z-10 size-[32rem] rounded-full bg-[radial-gradient(circle,var(--ds-glow-primary),transparent_70%)]"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 left-[-10rem] -z-10 size-80 rounded-full bg-[radial-gradient(circle,var(--ds-glow-secondary),transparent_70%)]"
        />
        <Container size="content">
          <div className="grid gap-12 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.62fr)] lg:items-end lg:gap-16">
            <Reveal>
              <div className="max-w-4xl">
                <Badge variant="accent">Contact</Badge>
                <h1 className="mt-7 text-balance font-serif text-5xl leading-[0.94] font-medium tracking-[-0.05em] text-ink sm:text-6xl lg:text-7xl">
                  Start a thoughtful conversation.
                </h1>
                <p className="mt-7 max-w-3xl text-pretty text-lg leading-8 text-ink/75 sm:text-xl sm:leading-9">
                  {contact.introduction}
                </p>
                <div className="mt-10 flex flex-wrap gap-3">
                  {emailMethod?.href ? (
                    <ButtonLink href={emailMethod.href}>
                      <Mail aria-hidden="true" className="size-4" />
                      Email {contact.name.split(" ")[0]}
                    </ButtonLink>
                  ) : null}
                  <ButtonLink href="/about" variant="secondary">
                    Read my profile
                    <ArrowRight aria-hidden="true" className="size-4" />
                  </ButtonLink>
                </div>
              </div>
            </Reveal>

            {contact.methods.length > 0 ? (
              <Reveal delay={0.08}>
                <Surface
                  as="aside"
                  aria-label="Available contact channels"
                  className="relative isolate overflow-hidden border-primary/20 bg-surface/80 shadow-[0_28px_90px_-58px_rgb(79_93_204/0.6)] backdrop-blur-sm"
                  padding="lg"
                  variant="raised"
                >
                  <span
                    aria-hidden="true"
                    className="absolute top-0 left-0 h-1 w-24 bg-gradient-to-r from-primary to-secondary"
                  />
                  <div className="flex items-end justify-between gap-5">
                    <div>
                      <p className="eyebrow text-primary">Connection map</p>
                      <h2 className="mt-4 font-serif text-2xl leading-tight font-medium tracking-[-0.03em] text-ink">
                        Available channels
                      </h2>
                    </div>
                    <span
                      aria-hidden="true"
                      className="font-serif text-4xl font-medium tracking-[-0.04em] text-primary"
                    >
                      {String(contact.methods.length).padStart(2, "0")}
                    </span>
                  </div>
                  <ul className="mt-7 divide-y divide-line border-y border-line">
                    {contact.methods.slice(0, 4).map((method) => {
                      const Icon = methodIcons[method.kind];

                      return (
                        <li key={method.key} className="flex items-center gap-3 py-4">
                          <span className="grid size-8 shrink-0 place-items-center rounded-full border border-line bg-canvas text-secondary">
                            <Icon aria-hidden="true" className="size-3.5" />
                          </span>
                          <span className="min-w-0 text-sm font-semibold text-ink">
                            {method.label}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </Surface>
              </Reveal>
            ) : null}
          </div>
        </Container>
      </section>

      <section id="contact-options" className="relative py-20 sm:py-24 lg:py-28">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent"
        />
        <Container size="content">
          <Reveal>
            <SectionHeading
              eyebrow="Contact options"
              title="Direct ways to connect."
              description={
                contact.methods.length > 0
                  ? "Published professional links provide direct ways to connect."
                  : "No contact methods are currently public."
              }
            />
          </Reveal>
          {contact.methods.length > 0 ? (
            <div className="mt-12 grid gap-5 md:grid-cols-2">
              {contact.methods.map((method, index) => (
                <Reveal
                  key={method.key}
                  className="h-full"
                  delay={(index % 2) * 0.05}
                >
                  <ContactMethodCard index={index} method={method} />
                </Reveal>
              ))}
            </div>
          ) : (
            <Reveal className="mt-12" delay={0.05}>
              <Surface padding="lg" variant="subtle">
                <p className="text-sm leading-7 text-muted">
                  No contact methods are currently public.
                </p>
              </Surface>
            </Reveal>
          )}
        </Container>
      </section>

      <section className="relative isolate overflow-hidden border-t border-line/80 bg-surface/55 py-16 sm:py-20">
        <div
          aria-hidden="true"
          className="foundation-grid pointer-events-none absolute inset-0 -z-10 opacity-25 [mask-image:linear-gradient(to_right,black,transparent_85%)]"
        />
        <Container size="content">
          <Reveal>
            <Surface
              variant="accent"
              padding="lg"
              className="relative overflow-hidden border-primary/20 shadow-[0_24px_72px_-58px_rgb(79_93_204/0.6)]"
            >
              <span
                aria-hidden="true"
                className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary to-secondary"
              />
              <div className="flex flex-col gap-7 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="eyebrow text-primary">Academic CV</p>
                  <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-ink">
                    Looking for a fuller professional overview?
                  </h2>
                  <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
                    Open the current Academic CV directly in a new browser tab.
                  </p>
                </div>
                <ButtonLink {...ACADEMIC_CV_LINK} variant="secondary">
                  View Academic CV
                  <ArrowRight aria-hidden="true" className="size-4" />
                </ButtonLink>
              </div>
            </Surface>
          </Reveal>
        </Container>
      </section>
    </main>
  );
}
