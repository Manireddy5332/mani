import type { ReactNode } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";

import { Badge, Container, Reveal, Surface } from "@/components/ui";

export type DetailIntroMetadataItem = {
  readonly label: string;
  readonly value: ReactNode;
};

export type DetailIntroProps = {
  readonly actions?: ReactNode;
  readonly currentLabel: string;
  readonly description: ReactNode;
  readonly eyebrow: ReactNode;
  readonly metadata: readonly DetailIntroMetadataItem[];
  readonly parentHref: `/${string}`;
  readonly parentLabel: string;
  readonly status?: ReactNode;
  readonly title: ReactNode;
};

/**
 * Shared editorial introduction for public record pages. Feature modules own
 * their record content while this component keeps breadcrumbs, hierarchy, and
 * metadata presentation consistent across detail routes.
 */
export function DetailIntro({
  actions,
  currentLabel,
  description,
  eyebrow,
  metadata,
  parentHref,
  parentLabel,
  status,
  title,
}: DetailIntroProps) {
  return (
    <section className="relative isolate overflow-hidden border-b border-line bg-canvas/45">
      <div
        aria-hidden="true"
        className="foundation-grid pointer-events-none absolute inset-0 opacity-70 [mask-image:radial-gradient(ellipse_88%_82%_at_45%_10%,black,transparent)]"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-40 -right-28 size-[28rem] rounded-full border border-primary/15 bg-primary/[0.03]"
      />
      <Container className="relative py-18 sm:py-22 lg:py-28">
        <Reveal>
          <nav aria-label="Breadcrumb">
            <ol className="flex flex-wrap items-center gap-2 font-mono text-[0.72rem] font-semibold tracking-[0.08em] text-subtle uppercase">
              <li>
                <Link
                  className="rounded-sm transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transition-none"
                  href="/"
                >
                  Home
                </Link>
              </li>
              <li aria-hidden="true">
                <ChevronRight className="size-3.5" />
              </li>
              <li>
                <Link
                  className="rounded-sm transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transition-none"
                  href={parentHref}
                >
                  {parentLabel}
                </Link>
              </li>
              <li aria-hidden="true">
                <ChevronRight className="size-3.5" />
              </li>
              <li>
                <span aria-current="page" className="text-ink">
                  {currentLabel}
                </span>
              </li>
            </ol>
          </nav>
        </Reveal>

        <div className="mt-12 grid items-end gap-12 lg:grid-cols-[minmax(0,1.2fr)_minmax(18rem,0.52fr)] lg:gap-16">
          <Reveal>
            <header className="max-w-5xl">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="accent">{eyebrow}</Badge>
                {status ? <Badge variant="outline">{status}</Badge> : null}
              </div>
              <h1 className="mt-8 text-balance font-serif text-[clamp(2.75rem,6vw,5.9rem)] leading-[0.93] font-medium tracking-[-0.055em] text-ink [overflow-wrap:anywhere]">
                {title}
              </h1>
              <div className="mt-8 max-w-3xl border-l-2 border-primary/55 pl-5 text-pretty text-base leading-8 text-muted sm:pl-7 sm:text-lg sm:leading-9">
                {description}
              </div>
              {actions ? (
                <div className="mt-9 flex flex-wrap gap-3">{actions}</div>
              ) : null}
            </header>
          </Reveal>

          <Reveal delay={0.08}>
            <Surface
              as="aside"
              aria-label={`${currentLabel} details`}
              className="relative overflow-hidden bg-surface/90 backdrop-blur-sm"
              padding="lg"
              variant="raised"
            >
              <span
                aria-hidden="true"
                className="absolute top-0 right-8 left-8 h-px bg-gradient-to-r from-transparent via-primary to-transparent"
              />
              <p className="eyebrow text-primary">At a glance</p>
              <dl className="mt-6 divide-y divide-line border-y border-line">
                {metadata.map((item) => (
                  <div key={item.label} className="grid gap-2 py-4 sm:grid-cols-[5.5rem_minmax(0,1fr)]">
                    <dt className="font-mono text-[0.68rem] font-semibold tracking-[0.1em] text-subtle uppercase">
                      {item.label}
                    </dt>
                    <dd className="text-sm leading-6 font-semibold text-ink [overflow-wrap:anywhere]">
                      {item.value}
                    </dd>
                  </div>
                ))}
              </dl>
            </Surface>
          </Reveal>
        </div>
      </Container>
    </section>
  );
}
