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
    <section className="relative overflow-hidden border-b border-line">
      <div
        aria-hidden="true"
        className="foundation-grid pointer-events-none absolute inset-0 opacity-65 [mask-image:linear-gradient(to_bottom,black,transparent_92%)]"
      />
      <Container className="relative py-18 sm:py-24 lg:py-28">
        <div className="grid items-end gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.6fr)] lg:gap-14">
          <Reveal>
            <header className="max-w-5xl">
              <Badge variant="accent">{eyebrow}</Badge>
              <h1 className="mt-7 text-balance font-serif text-[clamp(2.85rem,7vw,6.75rem)] leading-[0.92] font-medium tracking-[-0.055em] text-ink [overflow-wrap:anywhere]">
                {title}
              </h1>
              <div className="mt-7 max-w-3xl text-pretty text-base leading-8 text-muted sm:text-lg">
                {description}
              </div>
              {actions ? (
                <div className="mt-8 flex flex-wrap gap-3">{actions}</div>
              ) : null}
            </header>
          </Reveal>

          {aside ? <Reveal delay={0.08}>{aside}</Reveal> : null}
        </div>
      </Container>
    </section>
  );
}
