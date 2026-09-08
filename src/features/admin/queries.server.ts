import "server-only";

import {
  adminNestedResourceSchema,
  adminRecordIdSchema,
  adminResourceSchema,
} from "@/features/admin/schemas";
import {
  loadAdminNestedContent,
  loadAdminCollection,
  loadAdminDashboardData,
  loadAdminProfileExists,
  loadAdminRecord,
} from "@/features/admin/repository.server";
import type {
  AdminDashboardDto,
  AdminNestedContentDto,
  AdminNestedResourceKey,
  AdminRecordDto,
  AdminRecordSummaryDto,
  AdminResourceKey,
} from "@/features/admin/types";
import { assertAdmin } from "@/lib/auth/authorization.server";

export async function getAdminDashboard(): Promise<AdminDashboardDto> {
  await assertAdmin();
  return loadAdminDashboardData();
}

export async function getAdminProfileExists(): Promise<boolean> {
  await assertAdmin();
  return loadAdminProfileExists();
}

export async function listAdminRecords(
  resource: AdminResourceKey,
): Promise<AdminRecordSummaryDto[]> {
  await assertAdmin();
  const verifiedResource = adminResourceSchema.parse(resource);
  return loadAdminCollection(verifiedResource);
}

export async function getAdminRecord(
  resource: AdminResourceKey,
  id: string,
): Promise<AdminRecordDto | null> {
  await assertAdmin();
  const verifiedResource = adminResourceSchema.parse(resource);
  const verifiedId = adminRecordIdSchema.parse(id);
  return loadAdminRecord(verifiedResource, verifiedId);
}

export async function getAdminNestedContent(
  resource: AdminNestedResourceKey,
  parentId: string,
): Promise<AdminNestedContentDto | null> {
  await assertAdmin();
  const verifiedResource = adminNestedResourceSchema.parse(resource);
  const verifiedParentId = adminRecordIdSchema.parse(parentId);
  return loadAdminNestedContent(verifiedResource, verifiedParentId);
}
