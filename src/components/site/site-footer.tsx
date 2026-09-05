import Link from "next/link";

import { siteConfig } from "@/lib/site";

type SiteFooterProps = {
  className?: string;
};

export function SiteFooter({ className = "" }: SiteFooterProps) {
  const year = new Date().getFullYear();

  return (
    <footer className={`border-t border-line bg-canvas ${className}`}>
      <div className="mx-auto grid w-full max-w-[90rem] gap-8 px-5 py-10 sm:px-8 md:grid-cols-[1fr_auto] md:items-end lg:px-12">
        <div>
          <Link
            href="/"
            className="inline-flex rounded-sm text-base font-semibold tracking-[-0.01em] text-ink transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary motion-reduce:transition-none"
          >
            {siteConfig.name}
          </Link>
          <p className="mt-2 max-w-xl text-sm leading-6 text-ink/62">
            {siteConfig.description}
          </p>
        </div>

        <div className="md:text-right">
          <nav aria-label="Secondary navigation" className="mb-4 flex flex-wrap gap-x-5 gap-y-2 md:justify-end">
            <Link className="text-sm font-semibold text-ink transition-colors hover:text-primary" href="/resume">
              Resume
            </Link>
            <Link className="text-sm font-semibold text-ink transition-colors hover:text-primary" href="/contact">
              Contact
            </Link>
            <Link className="text-sm font-semibold text-ink transition-colors hover:text-primary" href="/about">
              About
            </Link>
          </nav>
          <p className="font-mono text-[0.68rem] uppercase tracking-[0.16em] text-ink/48">
            Academic + professional portfolio
          </p>
          <p className="mt-2 text-sm text-ink/62">
            © {year} {siteConfig.name}
          </p>
        </div>
      </div>
    </footer>
  );
}
