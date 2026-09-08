import { notFound } from "next/navigation";

import { AdminResourceList } from "@/components/admin/admin-resource-list";
import { getAdminResource } from "@/features/admin/registry";
import {
  getAdminProfileExists,
  listAdminRecords,
} from "@/features/admin/queries.server";
import { requireAdmin } from "@/lib/auth/authorization.server";

type AdminResourcePageProps = {
  params: Promise<{ resource: string }>;
};

export default async function AdminResourcePage({
  params,
}: AdminResourcePageProps) {
  await requireAdmin();
  const { resource: resourceParam } = await params;
  const resource = getAdminResource(resourceParam);

  if (!resource) notFound();

  const [records, profileExists] = await Promise.all([
    listAdminRecords(resource.key),
    getAdminProfileExists(),
  ]);

  return (
    <AdminResourceList
      profileExists={profileExists}
      records={records}
      resource={resource}
    />
  );
}
