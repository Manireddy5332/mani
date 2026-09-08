import type { Metadata } from "next";

import { GoogleSignInButton } from "@/components/admin/auth-buttons";
import { ButtonLink } from "@/components/ui/button";
import { Container } from "@/components/ui/container";
import { Surface } from "@/components/ui/surface";

export const metadata: Metadata = {
  title: "Administrator sign in",
  description: "Secure sign-in for the private portfolio administration area.",
  robots: { index: false, follow: false },
};

type SignInPageProps = {
  searchParams: Promise<{ error?: string | string[] }>;
};

export default async function SignInPage({ searchParams }: SignInPageProps) {
  const parameters = await searchParams;
  const hasOAuthError = typeof parameters.error !== "undefined";

  return (
    <main className="flex flex-1 items-center py-16 sm:py-24" id="main-content">
      <Container size="narrow">
        <Surface className="mx-auto max-w-xl" padding="lg" variant="raised">
          <p className="eyebrow text-primary">Private administration</p>
          <h1 className="mt-5 font-serif text-4xl leading-tight tracking-[-0.035em] text-ink sm:text-5xl">
            Sign in to manage the portfolio.
          </h1>
          <p className="mt-5 max-w-lg text-base leading-7 text-muted">
            This area is restricted to the verified administrator account.
            Continue with Google to establish a secure session.
          </p>

          {hasOAuthError ? (
            <p
              className="mt-5 rounded-xl border border-danger/30 bg-danger/[0.06] p-4 text-sm leading-6 text-danger"
              role="alert"
            >
              Authentication was not completed. Confirm the permitted Google
              account and try again.
            </p>
          ) : null}

          <div className="mt-8">
            <GoogleSignInButton />
          </div>

          <div className="mt-4 border-t border-line pt-6">
            <ButtonLink href="/" variant="ghost">
              Return to portfolio
            </ButtonLink>
          </div>
        </Surface>
      </Container>
    </main>
  );
}
