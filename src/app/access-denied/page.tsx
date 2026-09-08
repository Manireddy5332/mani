import type { Metadata } from "next";

import { SignOutButton } from "@/components/admin/auth-buttons";
import { ButtonLink } from "@/components/ui/button";
import { Container } from "@/components/ui/container";
import { Surface } from "@/components/ui/surface";

export const metadata: Metadata = {
  title: "Access denied",
  description: "The requested private administration area is not available.",
  robots: { index: false, follow: false },
};

export default function AccessDeniedPage() {
  return (
    <main className="flex flex-1 items-center py-16 sm:py-24" id="main-content">
      <Container size="narrow">
        <Surface className="mx-auto max-w-xl" padding="lg" variant="raised">
          <p className="eyebrow text-danger">Access denied</p>
          <h1 className="mt-5 font-serif text-4xl leading-tight tracking-[-0.035em] text-ink sm:text-5xl">
            This account cannot open the admin area.
          </h1>
          <p className="mt-5 max-w-lg text-base leading-7 text-muted">
            The portfolio is safe. No administrative content or controls were
            disclosed. Sign out before trying the authorized account.
          </p>

          <div className="mt-8 flex flex-wrap items-start gap-4">
            <SignOutButton />
            <ButtonLink href="/" variant="ghost">
              Return to portfolio
            </ButtonLink>
          </div>
        </Surface>
      </Container>
    </main>
  );
}
