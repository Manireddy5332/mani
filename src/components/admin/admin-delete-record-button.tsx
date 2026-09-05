"use client";

import { Archive } from "lucide-react";
import { useRouter } from "next/navigation";
import { useActionState, useEffect } from "react";

import { Button } from "@/components/ui";
import { deleteAdminRecordAction } from "@/features/admin/actions";
import type {
  AdminActionResult,
  AdminResourceKey,
} from "@/features/admin/types";

const initialState: AdminActionResult = { ok: false, message: "" };

type AdminDeleteRecordButtonProps = {
  id: string;
  resource: AdminResourceKey;
  title: string;
};

export function AdminDeleteRecordButton({
  id,
  resource,
  title,
}: AdminDeleteRecordButtonProps) {
  const router = useRouter();
  const [state, formAction, pending] = useActionState(
    async (): Promise<AdminActionResult> =>
      deleteAdminRecordAction(resource, id, { confirmed: true }),
    initialState,
  );

  useEffect(() => {
    if (state.ok) router.refresh();
  }, [router, state.ok]);

  return (
    <form
      action={formAction}
      onSubmit={(event) => {
        const confirmed = window.confirm(
          `Archive “${title}”? It will remain in the database for provenance and can be edited later.`,
        );

        if (!confirmed) event.preventDefault();
      }}
      className="inline-flex flex-col items-end"
    >
      <Button
        type="submit"
        disabled={pending}
        size="sm"
        variant="ghost"
        className="text-danger hover:text-danger"
        aria-label={`Archive ${title}`}
      >
        <Archive aria-hidden="true" className="size-4" />
        {pending ? "Archiving…" : "Archive"}
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
