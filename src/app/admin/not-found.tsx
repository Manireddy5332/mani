import { SearchX } from "lucide-react";

import { AdminPageHeader } from "@/components/admin/admin-page-header";
import { ButtonLink, Surface } from "@/components/ui";

export default function AdminNotFound() {
  return (
    <div>
      <AdminPageHeader
        eyebrow="Not found"
        title="That admin record is not available."
        description="The resource is not part of the approved admin allowlist, or the requested record no longer exists."
      />
      <Surface className="mt-8 text-center" padding="lg" variant="subtle">
        <SearchX aria-hidden="true" className="mx-auto size-8 text-primary" />
        <p className="mx-auto mt-4 max-w-lg text-sm leading-7 text-muted">
          Return to the admin overview to choose an available content area.
        </p>
        <div className="mt-6">
          <ButtonLink href="/admin">Admin overview</ButtonLink>
        </div>
      </Surface>
    </div>
  );
}
