"use server";

import { revalidatePath, updateTag } from "next/cache";
import { ZodError } from "zod";

import { Prisma } from "@/generated/prisma/client";
import {
  adminCreateInputSchemas,
  adminDeleteInputSchema,
  adminNestedResourceSchema,
  adminRecordIdSchema,
  adminResourceSchema,
  adminUpdateInputSchemas,
  adminVisibilitySchema,
  parseAdminCreateInput,
  parseAdminNestedContentInput,
  parseAdminUpdateInput,
} from "@/features/admin/schemas";
import { getAdminInvalidationPlan } from "@/features/admin/policies";
import {
  AdminRepositoryError,
  createAdminRecord,
  deleteAdminRecord,
  saveAdminNestedContent,
  setAdminRecordVisibility,
  updateAdminRecord,
} from "@/features/admin/repository.server";
import type {
  AdminActionResult,
  AdminNestedActionResult,
  AdminNestedResourceKey,
  AdminResourceKey,
  AdminVisibilityStatus,
} from "@/features/admin/types";
import { assertAdmin } from "@/lib/auth/authorization.server";
import { clearPublicLastKnownGood } from "@/lib/public-content/fallback.server";

function validationFailure(error: ZodError): AdminActionResult {
  const fieldErrors: Record<string, string[]> = {};
  for (const issue of error.issues) {
    const field = issue.path.join(".") || "form";
    (fieldErrors[field] ??= []).push(issue.message);
  }
  return {
    ok: false,
    message: "Review the highlighted fields and try again.",
    fieldErrors,
  };
}

function nestedSafeFailure(error: unknown): AdminNestedActionResult {
  const failure = safeFailure(error);
  return {
    ok: false,
    message: failure.message,
    ...(failure.fieldErrors ? { fieldErrors: failure.fieldErrors } : {}),
  };
}

function safeFailure(error: unknown): AdminActionResult {
  if (error instanceof ZodError) return validationFailure(error);
  if (error instanceof AdminRepositoryError) {
    return { ok: false, message: error.message };
  }
  if (error instanceof Prisma.PrismaClientKnownRequestError) {
    if (error.code === "P2002") {
      return {
        ok: false,
        message: "A record already uses one of these unique values.",
      };
    }
    if (error.code === "P2003" || error.code === "P2014") {
      return {
        ok: false,
        message: "This record is still referenced by other portfolio content.",
      };
    }
    if (error.code === "P2004") {
      return {
        ok: false,
        message: "The submitted values violate a portfolio data constraint.",
      };
    }
    if (error.code === "P2025") {
      return { ok: false, message: "The requested record was not found." };
    }
  }

  return {
    ok: false,
    message: "The change could not be saved. Please try again.",
  };
}

function revalidateAdminResource(resource: AdminResourceKey, id?: string) {
  const plan = getAdminInvalidationPlan(resource, id);
  clearPublicLastKnownGood(plan.tags);
  for (const tag of plan.tags) updateTag(tag);

  revalidatePath("/admin", "layout");
  for (const path of plan.adminPaths) revalidatePath(path);
  for (const path of plan.publicPaths) revalidatePath(path);
}

export async function createAdminRecordAction(
  resource: AdminResourceKey,
  input: unknown,
): Promise<AdminActionResult> {
  await assertAdmin();

  try {
    const verifiedResource = adminResourceSchema.parse(resource);
    if (verifiedResource === "site-settings") {
      adminCreateInputSchemas[verifiedResource].parse(input);
    }
    const data = parseAdminCreateInput(verifiedResource, input);
    const record = await createAdminRecord(verifiedResource, data);
    revalidateAdminResource(verifiedResource, record.id);
    return { ok: true, message: "Record created.", record };
  } catch (error) {
    return safeFailure(error);
  }
}

export async function updateAdminRecordAction(
  resource: AdminResourceKey,
  id: string,
  input: unknown,
): Promise<AdminActionResult> {
  await assertAdmin();

  try {
    const verifiedResource = adminResourceSchema.parse(resource);
    const verifiedId = adminRecordIdSchema.parse(id);
    if (verifiedResource === "site-settings") {
      adminUpdateInputSchemas[verifiedResource].parse(input);
    }
    const data = parseAdminUpdateInput(verifiedResource, input);
    const record = await updateAdminRecord(verifiedResource, verifiedId, data);
    revalidateAdminResource(verifiedResource, record.id);
    return { ok: true, message: "Changes saved.", record };
  } catch (error) {
    return safeFailure(error);
  }
}

export async function deleteAdminRecordAction(
  resource: AdminResourceKey,
  id: string,
  confirmation: unknown,
): Promise<AdminActionResult> {
  await assertAdmin();

  try {
    const verifiedResource = adminResourceSchema.parse(resource);
    const verifiedId = adminRecordIdSchema.parse(id);
    adminDeleteInputSchema.parse(confirmation);
    await deleteAdminRecord(verifiedResource, verifiedId);
    revalidateAdminResource(verifiedResource);
    return {
      ok: true,
      message: "Record archived.",
    };
  } catch (error) {
    return safeFailure(error);
  }
}

export async function setAdminRecordVisibilityAction(
  resource: AdminResourceKey,
  id: string,
  visibility: AdminVisibilityStatus,
): Promise<AdminActionResult> {
  await assertAdmin();

  try {
    const verifiedResource = adminResourceSchema.parse(resource);
    const verifiedId = adminRecordIdSchema.parse(id);
    const verifiedVisibility = adminVisibilitySchema.parse(visibility);
    if (verifiedResource === "site-settings") {
      throw new AdminRepositoryError(
        "READ_ONLY",
        "Site settings do not support visibility changes.",
      );
    }
    const record = await setAdminRecordVisibility(
      verifiedResource,
      verifiedId,
      verifiedVisibility,
    );
    revalidateAdminResource(verifiedResource, verifiedId);
    return {
      ok: true,
      message:
        verifiedVisibility === "PUBLISHED"
          ? "Record published."
          : "Record unpublished.",
      record,
    };
  } catch (error) {
    return safeFailure(error);
  }
}

export async function saveAdminNestedContentAction(
  resource: AdminNestedResourceKey,
  parentId: string,
  input: unknown,
): Promise<AdminNestedActionResult> {
  await assertAdmin();

  try {
    const verifiedResource = adminNestedResourceSchema.parse(resource);
    const verifiedParentId = adminRecordIdSchema.parse(parentId);
    const data = parseAdminNestedContentInput(verifiedResource, input);
    const outcome = await saveAdminNestedContent(
      verifiedResource,
      verifiedParentId,
      data,
    );
    if (outcome.kind === "confirmation-required") {
      return {
        ok: false,
        message: "Confirm the listed removals before saving.",
        confirmation: outcome.confirmation,
      };
    }

    revalidateAdminResource(verifiedResource, verifiedParentId);
    return {
      ok: true,
      message: "Nested content saved.",
      content: outcome.content,
    };
  } catch (error) {
    return nestedSafeFailure(error);
  }
}
