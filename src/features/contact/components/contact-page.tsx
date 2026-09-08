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
  SectionHeading,
  Surface,
} from "@/components/ui";

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

function ContactMethodCard({ method }: { readonly method: ContactMethod }) {
  const Icon = methodIcons[method.kind];

  return (
    <Surface as="article" padding="lg" className="h-full">
      <div className="grid size-10 place-items-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
        <Icon aria-hidden="true" className="size-5" />
      </div>
      <h3 className="mt-6 eyebrow text-muted">{method.label}</h3>
      {method.href ? (
        <a
          className="mt-3 inline-flex max-w-full items-center gap-2 break-all text-lg font-semibold tracking-[-0.02em] text-ink underline decoration-line underline-offset-4 transition-colors hover:text-primary"
          href={method.href}
          rel={method.external ? "noreferrer" : undefined}
          target={method.external ? "_blank" : undefined}
        >
          {method.value}
          {method.external ? (
            <ArrowUpRight aria-hidden="true" className="size-4 shrink-0" />
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
    <main id="main-content">
      <section className="foundation-grid border-b border-line/80 py-20 sm:py-24 lg:py-28">
        <Container size="content">
          <div className="max-w-4xl">
            <Badge variant="accent">Contact</Badge>
            <h1 className="mt-7 text-balance font-serif text-5xl leading-[0.98] font-medium tracking-[-0.045em] text-ink sm:text-6xl lg:text-7xl">
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
        </Container>
      </section>

      <section id="contact-options" className="py-20 sm:py-24">
        <Container size="content">
          <SectionHeading
            eyebrow="Contact options"
            title="Direct ways to connect."
            description={
              contact.methods.length > 0
                ? "Published professional links provide direct ways to connect."
                : "No contact methods are currently public."
            }
          />
          {contact.methods.length > 0 ? (
            <div className="mt-12 grid gap-5 md:grid-cols-2">
              {contact.methods.map((method) => (
                <ContactMethodCard key={method.key} method={method} />
              ))}
            </div>
          ) : (
            <Surface className="mt-12" padding="lg" variant="subtle">
              <p className="text-sm leading-7 text-muted">
                No contact methods are currently public.
              </p>
            </Surface>
          )}
        </Container>
      </section>

      <section className="border-t border-line/80 bg-surface/55 py-16 sm:py-20">
        <Container size="content">
          <Surface
            variant="accent"
            padding="lg"
            className="flex flex-col gap-7 sm:flex-row sm:items-center sm:justify-between"
          >
            <div>
              <p className="eyebrow text-primary">Academic CV</p>
              <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-ink">
                Looking for a fuller professional overview?
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
                Review the on-page CV summary or request the current document by
                email.
              </p>
            </div>
            <ButtonLink href="/resume" variant="secondary">
              View CV overview
              <ArrowRight aria-hidden="true" className="size-4" />
            </ButtonLink>
          </Surface>
        </Container>
      </section>
    </main>
  );
}
