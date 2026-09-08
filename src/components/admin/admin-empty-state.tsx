import { Inbox } from "lucide-react";

import { ButtonLink, Surface } from "@/components/ui";

type AdminEmptyStateProps = {
  actionHref?: string;
  actionLabel?: string;
  description: string;
  title: string;
};

export function AdminEmptyState({
  actionHref,
  actionLabel,
  description,
  title,
}: AdminEmptyStateProps) {
  return (
    <Surface className="text-center" padding="lg" variant="subtle">
      <span className="mx-auto grid size-12 place-items-center rounded-2xl border border-line bg-canvas text-primary">
        <Inbox aria-hidden="true" className="size-5" />
      </span>
      <h2 className="mt-5 font-serif text-2xl font-medium tracking-[-0.025em] text-ink">
        {title}
      </h2>
      <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-muted">
        {description}
      </p>
      {actionHref && actionLabel ? (
        <div className="mt-6">
          <ButtonLink href={actionHref} size="sm">
            {actionLabel}
          </ButtonLink>
        </div>
      ) : null}
    </Surface>
  );
}
