import Link from "next/link";

import { PrimaryNavigation } from "@/components/site/primary-navigation";
import { ThemeToggle } from "@/components/site/theme-toggle";
import { siteConfig } from "@/lib/site";

type SiteHeaderProps = {
  className?: string;
};

export function SiteHeader({ className = "" }: SiteHeaderProps) {
  return (
    <header
      className={`sticky top-0 z-50 overflow-x-clip border-b border-line/80 bg-canvas/90 backdrop-blur-xl ${className}`}
    >
      <div className="mx-auto flex w-full max-w-[90rem] flex-col px-5 sm:px-8 lg:flex-row lg:items-center lg:gap-10 lg:px-12">
        <div className="flex min-h-20 items-center justify-between gap-5">
          <Link
            href="/"
            className="group inline-flex min-w-0 items-center gap-3 rounded-sm focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
            aria-label={`${siteConfig.brandName}, home`}
          >
            <span
              aria-hidden="true"
              className="grid size-10 shrink-0 place-items-center border border-ink bg-ink font-mono text-[0.68rem] font-semibold tracking-[0.16em] text-canvas transition-colors duration-200 group-hover:border-primary group-hover:bg-primary group-hover:text-primary-contrast motion-reduce:transition-none"
            >
              {siteConfig.shortName}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold tracking-[-0.01em] text-ink">
                {siteConfig.brandName}
              </span>
              <span className="mt-0.5 block truncate font-mono text-[0.68rem] uppercase tracking-[0.14em] text-muted">
                {siteConfig.role}
              </span>
            </span>
          </Link>

          <div className="lg:hidden">
            <ThemeToggle />
          </div>
        </div>

        <PrimaryNavigation />

        <div className="ml-2 hidden lg:block">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
