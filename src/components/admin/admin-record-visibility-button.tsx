"use client";

import { Eye, EyeOff } from "lucide-react";
import { useRouter } from "next/navigation";
import { useActionState, useEffect } from "react";

import { Button } from "@/components/ui";
import { setAdminRecordVisibilityAction } from "@/features/admin/actions";
import type {
  AdminActionResult,
  AdminResourceKey,
} from "@/features/admin/types";

const initialState: AdminActionResult = { ok: false, message: "" };

type AdminRecordVisibilityButtonProps = {
  id: string;
  resource: AdminResourceKey;
  status: "DRAFT" | "PUBLISHED";
  title: string;
};

export function AdminRecordVisibilityButton({
  id,
  resource,
  status,
  title,
}: AdminRecordVisibilityButtonProps) {
  const router = useRouter();
  const nextVisibility = status === "PUBLISHED" ? "DRAFT" : "PUBLISHED";
  const unpublishing = nextVisibility === "DRAFT";
  const [state, formAction, pending] = useActionState(
    async (): Promise<AdminActionResult> =>
      setAdminRecordVisibilityAction(resource, id, nextVisibility),
    initialState,
  );
  const refreshMarker = state.ok ? state.record?.updatedAt : undefined;

  useEffect(() => {
    if (refreshMarker) router.refresh();
  }, [refreshMarker, router]);

  return (
    <form
      action={formAction}
      className="inline-flex flex-col items-end"
      onSubmit={(event) => {
        if (!unpublishing) return;

        const confirmed = window.confirm(
          `Unpublish “${title}”? It will be removed from public portfolio views but remain available in the admin area.`,
        );

        if (!confirmed) event.preventDefault();
      }}
    >
      <Button
        type="submit"
        disabled={pending}
        size="sm"
        variant="ghost"
        aria-label={`${unpublishing ? "Unpublish" : "Publish"} ${title}`}
      >
        {unpublishing ? (
          <EyeOff aria-hidden="true" className="size-4" />
        ) : (
          <Eye aria-hidden="true" className="size-4" />
        )}
        {pending
          ? unpublishing
            ? "Unpublishing…"
            : "Publishing…"
          : unpublishing
            ? "Unpublish"
            : "Publish"}
      </Button>
      {state.message ? (
        <span
          aria-live="polite"
          className={`mt-1 max-w-52 text-right text-xs ${state.ok ? "text-secondary" : "text-danger"}`}
          role={state.ok ? "status" : "alert"}
        >
          {state.message}
        </span>
      ) : null}
    </form>
  );
}
