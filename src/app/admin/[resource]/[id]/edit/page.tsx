import { notFound } from "next/navigation";

import { AdminNestedContentEditor } from "@/components/admin/admin-nested-content-editor";
import { AdminPageHeader } from "@/components/admin/admin-page-header";
import { AdminResourceForm } from "@/components/admin/admin-resource-form";
import {
  getAdminRelationResources,
  getAdminResource,
} from "@/features/admin/registry";
import {
  getAdminNestedContent,
  getAdminRecord,
  listAdminRecords,
} from "@/features/admin/queries.server";
import { adminRecordIdSchema } from "@/features/admin/schemas";
import type { AdminRelationOptions } from "@/features/admin/types";
import { requireAdmin } from "@/lib/auth/authorization.server";

type EditAdminRecordPageProps = {
  params: Promise<{ id: string; resource: string }>;
};

export default async function EditAdminRecordPage({
  params,
}: EditAdminRecordPageProps) {
  await requireAdmin();
  const { id, resource: resourceParam } = await params;
  const resource = getAdminResource(resourceParam);

  if (!resource?.capabilities.update || !adminRecordIdSchema.safeParse(id).success) {
    notFound();
  }

  const nestedResource =
    resource.key === "research-projects" || resource.key === "projects"
      ? resource.key
      : null;

  const [record, nestedContent, ...relationRecords] = await Promise.all([
    getAdminRecord(resource.key, id),
    nestedResource
      ? getAdminNestedContent(nestedResource, id)
      : Promise.resolve(null),
    ...getAdminRelationResources(resource).map(async (key) =>
      (await listAdminRecords(key)).filter(
        (relation) => relation.status !== "ARCHIVED",
      ),
    ),
  ]);

  if (!record) notFound();
  if (nestedResource && !nestedContent) notFound();

  const relationOptions = Object.fromEntries(
    getAdminRelationResources(resource).map((key, index) => [
      key,
      relationRecords[index] ?? [],
    ]),
  ) as AdminRelationOptions;

  return (
    <div>
      <AdminPageHeader
        eyebrow={`Edit ${resource.singularLabel}`}
        title={record.title}
        description="Update only details that remain accurate and verified. Saving an existing CV-backed record marks it as administrator-managed so later seed runs do not overwrite it."
        backHref={`/admin/${resource.key}`}
        backLabel={`Back to ${resource.label.toLowerCase()}`}
      />
      <div className="mt-8">
        <AdminResourceForm
          mode="edit"
          record={record}
          relationOptions={relationOptions}
          resource={resource}
        />
      </div>
      {nestedContent ? (
        <div className="mt-14 border-t border-line pt-10">
          <AdminNestedContentEditor content={nestedContent} />
        </div>
      ) : null}
    </div>
  );
}
