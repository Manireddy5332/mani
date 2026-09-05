"use client";

import { LoaderCircle, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";
import { performSecureSignOut } from "@/lib/auth/sign-out";

export function GoogleSignInButton() {
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function signIn() {
    setError(null);
    setIsPending(true);

    try {
      const result = await authClient.signIn.social({
        provider: "google",
        callbackURL: "/admin",
        errorCallbackURL: "/sign-in?error=oauth",
      });

      if (result.error) {
        setError("Google sign-in could not be completed. Please try again.");
        setIsPending(false);
      }
    } catch {
      setError("Google sign-in could not be completed. Please try again.");
      setIsPending(false);
    }
  }

  return (
    <div className="space-y-3">
      <Button
        aria-describedby={error ? "google-sign-in-error" : undefined}
        disabled={isPending}
        onClick={signIn}
        size="lg"
        wide
      >
        {isPending ? (
          <LoaderCircle
            aria-hidden="true"
            className="size-4 animate-spin motion-reduce:animate-none"
          />
        ) : (
          <span
            aria-hidden="true"
            className="grid size-5 place-items-center rounded-full bg-primary-contrast font-sans text-xs font-bold text-primary"
          >
            G
          </span>
        )}
        {isPending ? "Opening Google..." : "Continue with Google"}
      </Button>
      <p
        aria-live="polite"
        className="min-h-5 text-sm text-danger"
        id="google-sign-in-error"
      >
        {error}
      </p>
    </div>
  );
}

export function SignOutButton() {
  const router = useRouter();
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function signOut() {
    setError(null);
    setIsPending(true);

    try {
      const result = await performSecureSignOut(authClient);

      if (!result.ok) {
        setError("Sign-out could not be completed. Please try again.");
        setIsPending(false);
        return;
      }

      router.replace("/sign-in");
    } catch {
      setError("Sign-out could not be completed. Please try again.");
      setIsPending(false);
    }
  }

  return (
    <div className="space-y-2">
      <Button
        aria-describedby={error ? "sign-out-error" : undefined}
        disabled={isPending}
        onClick={signOut}
        size="sm"
        variant="outline"
      >
        {isPending ? (
          <LoaderCircle
            aria-hidden="true"
            className="size-4 animate-spin motion-reduce:animate-none"
          />
        ) : (
          <LogOut aria-hidden="true" className="size-4" />
        )}
        {isPending ? "Signing out..." : "Sign out"}
      </Button>
      <p
        aria-live="polite"
        className="max-w-64 text-xs text-danger"
        id="sign-out-error"
      >
        {error}
      </p>
    </div>
  );
}
