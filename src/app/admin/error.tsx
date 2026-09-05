"use client";

import { AlertTriangle, RotateCcw } from "lucide-react";

import { AdminPageHeader } from "@/components/admin/admin-page-header";
import { Button, Surface } from "@/components/ui";

type AdminErrorProps = {
  error: Error & { digest?: string };
  unstable_retry: () => void;
};

export default function AdminError({ unstable_retry }: AdminErrorProps) {
  return (
    <div>
      <AdminPageHeader
        eyebrow="Admin error"
        title="This admin view could not be loaded."
        description="No content was changed. Try the request again; if the problem continues, verify the local database and authentication configuration."
      />
      <Surface className="mt-8 text-center" padding="lg" variant="subtle">
        <AlertTriangle aria-hidden="true" className="mx-auto size-8 text-danger" />
        <p className="mx-auto mt-4 max-w-xl text-sm leading-7 text-muted">
          Technical error details are intentionally not displayed in the browser.
        </p>
        <div className="mt-6">
          <Button onClick={() => unstable_retry()}>
            <RotateCcw aria-hidden="true" className="size-4" />
            Try again
          </Button>
        </div>
      </Surface>
    </div>
  );
}
