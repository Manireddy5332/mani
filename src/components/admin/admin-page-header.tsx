import { ChevronLeft } from "lucide-react";
import Link from "next/link";

type AdminPageHeaderProps = {
  actions?: React.ReactNode;
  backHref?: string;
  backLabel?: string;
  description: React.ReactNode;
  eyebrow?: string;
  title: React.ReactNode;
};

export function AdminPageHeader({
  actions,
  backHref,
  backLabel = "Back",
  description,
  eyebrow = "Administration",
  title,
}: AdminPageHeaderProps) {
  return (
    <header className="border-b border-line pb-7">
      {backHref ? (
        <Link
          href={backHref}
          className="mb-5 inline-flex items-center gap-1.5 rounded-sm text-sm font-semibold text-muted transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
        >
          <ChevronLeft aria-hidden="true" className="size-4" />
          {backLabel}
        </Link>
      ) : null}
      <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        <div className="max-w-3xl">
          <p className="font-mono text-[0.7rem] font-semibold tracking-[0.16em] text-primary uppercase">
            {eyebrow}
          </p>
          <h1 className="mt-3 text-balance font-serif text-4xl leading-[1.02] font-medium tracking-[-0.04em] text-ink sm:text-5xl">
            {title}
          </h1>
          <div className="mt-4 max-w-2xl text-sm leading-7 text-muted sm:text-base">
            {description}
          </div>
        </div>
        {actions ? (
          <div className="flex shrink-0 flex-wrap gap-3">{actions}</div>
        ) : null}
      </div>
    </header>
  );
}
