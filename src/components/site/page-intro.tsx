import type { ReactNode } from "react";

import { Badge, Container, Reveal } from "@/components/ui";

export type PageIntroProps = {
  actions?: ReactNode;
  aside?: ReactNode;
  description: ReactNode;
  eyebrow: ReactNode;
  title: ReactNode;
};

/**
 * Shared editorial introduction for public index pages. Route files remain
 * thin while feature components keep a consistent visual and semantic entry.
 */
export function PageIntro({
  actions,
  aside,
  description,
  eyebrow,
  title,
}: PageIntroProps) {
  return (
    <section className="relative isolate overflow-hidden border-b border-line bg-canvas/45">
      <div
        aria-hidden="true"
        className="foundation-grid pointer-events-none absolute inset-0 opacity-70 [mask-image:radial-gradient(ellipse_85%_80%_at_50%_12%,black,transparent)]"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 right-[7%] size-64 rounded-full border border-primary/15 bg-primary/[0.035] sm:size-96"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-20 right-[18%] size-3 rounded-full bg-secondary shadow-[0_0_0_10px_color-mix(in_srgb,var(--ds-secondary)_12%,transparent)]"
      />
      <Container className="relative py-20 sm:py-26 lg:py-32">
        <div className="grid items-end gap-12 lg:grid-cols-[minmax(0,1.22fr)_minmax(18rem,0.58fr)] lg:gap-16">
          <Reveal>
            <header className="max-w-5xl">
              <Badge variant="accent">{eyebrow}</Badge>
              <h1 className="mt-8 text-balance font-serif text-[clamp(3rem,7vw,6.9rem)] leading-[0.9] font-medium tracking-[-0.058em] text-ink [overflow-wrap:anywhere]">
                {title}
              </h1>
              <div className="mt-8 max-w-3xl border-l-2 border-primary/55 pl-5 text-pretty text-base leading-8 text-muted sm:pl-7 sm:text-lg">
                {description}
              </div>
              {actions ? (
                <div className="mt-9 flex flex-wrap gap-3">{actions}</div>
              ) : null}
            </header>
          </Reveal>

          {aside ? (
            <Reveal className="relative lg:pl-8" delay={0.08}>
              <span
                aria-hidden="true"
                className="absolute top-3 bottom-3 left-0 hidden w-px bg-gradient-to-b from-primary/70 via-line to-transparent lg:block"
              />
              {aside}
            </Reveal>
          ) : null}
        </div>
      </Container>
    </section>
  );
}
