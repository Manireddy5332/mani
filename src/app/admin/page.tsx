import { AdminDashboard } from "@/components/admin/admin-dashboard";
import { getAdminDashboard } from "@/features/admin/queries.server";
import { requireAdmin } from "@/lib/auth/authorization.server";

export default async function AdminPage() {
  await requireAdmin();
  const dashboard = await getAdminDashboard();

  return <AdminDashboard dashboard={dashboard} />;
}
