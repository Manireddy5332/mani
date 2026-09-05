import { notFound, redirect } from "next/navigation";

import { AdminPageHeader } from "@/components/admin/admin-page-header";
import { AdminResourceForm } from "@/components/admin/admin-resource-form";
import {
  getAdminRelationResources,
  getAdminResource,
} from "@/features/admin/registry";
import {
  getAdminProfileExists,
  listAdminRecords,
} from "@/features/admin/queries.server";
import type { AdminRelationOptions } from "@/features/admin/types";
import { requireAdmin } from "@/lib/auth/authorization.server";

type NewAdminRecordPageProps = {
  params: Promise<{ resource: string }>;
};

export default async function NewAdminRecordPage({
  params,
}: NewAdminRecordPageProps) {
  await requireAdmin();
  const { resource: resourceParam } = await params;
  const resource = getAdminResource(resourceParam);

  if (!resource?.capabilities.create) notFound();

  const profileExists = await getAdminProfileExists();
  if (resource.key === "profile" && profileExists) {
    redirect("/admin/profile");
  }
  if (
    resource.key !== "profile" &&
    resource.key !== "site-settings" &&
    !profileExists
  ) {
    redirect(`/admin/${resource.key}`);
  }

  const relationResources = getAdminRelationResources(resource);
  const relationEntries = await Promise.all(
    relationResources.map(
      async (key) => [
        key,
        (await listAdminRecords(key)).filter(
          (record) => record.status !== "ARCHIVED",
        ),
      ] as const,
    ),
  );
  const relationOptions = Object.fromEntries(
    relationEntries,
  ) as AdminRelationOptions;

  return (
    <div>
      <AdminPageHeader
        eyebrow={`New ${resource.singularLabel}`}
        title={`Create ${resource.singularLabel}`}
        description="Start with verified information only. New records default to a private draft unless you explicitly choose another visibility state."
        backHref={`/admin/${resource.key}`}
        backLabel={`Back to ${resource.label.toLowerCase()}`}
      />
      <div className="mt-8">
        <AdminResourceForm
          mode="create"
          relationOptions={relationOptions}
          resource={resource}
        />
      </div>
    </div>
  );
}
