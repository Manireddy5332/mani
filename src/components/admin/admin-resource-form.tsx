"use client";

import { AlertTriangle, Save } from "lucide-react";
import { useRouter } from "next/navigation";
import { useActionState, useEffect, useId, useRef } from "react";

import { Button, ButtonLink } from "@/components/ui";
import {
  createAdminRecordAction,
  updateAdminRecordAction,
} from "@/features/admin/actions";
import type {
  AdminActionResult,
  AdminFieldDefinition,
  AdminFieldValue,
  AdminFormMode,
  AdminRecordDto,
  AdminRelationOptions,
  AdminResourceDefinition,
} from "@/features/admin/types";
import { cn } from "@/lib/cn";

const initialActionState: AdminActionResult = { ok: false, message: "" };

const controlStyles =
  "min-h-11 w-full rounded-xl border border-line bg-canvas px-3.5 py-2.5 text-sm text-ink shadow-[0_1px_0_rgb(20_25_35/0.02)] transition-[border-color,box-shadow] placeholder:text-muted/65 focus:border-primary focus:outline-none focus:ring-3 focus:ring-primary/10 disabled:cursor-not-allowed disabled:opacity-55";

type AdminResourceFormProps = {
  mode: AdminFormMode;
  record?: AdminRecordDto;
  relationOptions?: AdminRelationOptions;
  resource: AdminResourceDefinition;
};

function serializeForm(
  fields: readonly AdminFieldDefinition[],
  formData: FormData,
): Record<string, AdminFieldValue> {
  return Object.fromEntries(
    fields.map((field) => {
      if (field.kind === "checkbox") {
        return [field.name, formData.get(field.name) === "on"];
      }

      const raw = formData.get(field.name);
      const value = typeof raw === "string" ? raw : "";

      if (field.kind === "number") {
        return [field.name, value.trim() === "" ? null : Number(value)];
      }

      if (field.kind === "string-list") {
        return [
          field.name,
          value
            .split(/\r?\n/)
            .map((item) => item.trim())
            .filter(Boolean),
        ];
      }

      return [field.name, value];
    }),
  );
}

function getDefaultValue(
  record: AdminRecordDto | undefined,
  field: AdminFieldDefinition,
): string | number | undefined {
  const value = record?.values[field.name];

  if (Array.isArray(value)) return value.join("\n");
  if (typeof value === "string" || typeof value === "number") return value;

  if (!record) {
    if (field.name === "status") return "DRAFT";
    if (field.name === "contentFormat") return "markdown";
    if (field.name === "sortOrder") return 0;
  }

  return undefined;
}

function AdminFormField({
  field,
  fieldErrors,
  record,
  relationOptions,
}: {
  field: AdminFieldDefinition;
  fieldErrors?: string[];
  record?: AdminRecordDto;
  relationOptions: AdminRelationOptions;
}) {
  const generatedId = useId();
  const id = `admin-${field.name}-${generatedId.replaceAll(":", "")}`;
  const descriptionId =
    field.description || field.kind === "string-list"
      ? `${id}-description`
      : undefined;
  const errorId = fieldErrors?.length ? `${id}-error` : undefined;
  const describedBy = [descriptionId, errorId].filter(Boolean).join(" ") || undefined;
  const defaultValue = getDefaultValue(record, field);
  const errorStyles = fieldErrors?.length
    ? "border-danger focus:border-danger focus:ring-danger/10"
    : undefined;

  if (field.kind === "checkbox") {
    return (
      <div className={cn("col-span-full", !field.fullWidth && "sm:col-span-1")}>
        <label
          htmlFor={id}
          className="flex min-h-12 cursor-pointer items-start gap-3 rounded-xl border border-line bg-canvas p-3.5"
        >
          <input
            id={id}
            name={field.name}
            type="checkbox"
            defaultChecked={record?.values[field.name] === true}
            aria-describedby={describedBy}
            aria-invalid={fieldErrors?.length ? true : undefined}
            className="mt-0.5 size-4 shrink-0 accent-primary"
          />
          <span>
            <span className="block text-sm font-semibold text-ink">{field.label}</span>
            {field.description ? (
              <span id={descriptionId} className="mt-1 block text-xs leading-5 text-muted">
                {field.description}
              </span>
            ) : null}
          </span>
        </label>
        {fieldErrors?.length ? (
          <p id={errorId} className="mt-2 text-xs leading-5 text-danger" role="alert">
            {fieldErrors.join(" ")}
          </p>
        ) : null}
      </div>
    );
  }

  const options =
    field.kind === "relation" && field.optionsResource
      ? (relationOptions[field.optionsResource] ?? []).map((item) => ({
          label: item.title,
          value: item.id,
        }))
      : field.options;

  return (
    <div className={cn("col-span-full", !field.fullWidth && "sm:col-span-1")}>
      <label htmlFor={id} className="block text-sm font-semibold text-ink">
        {field.label}
        {field.required ? <span className="ml-1 text-danger" aria-hidden="true">*</span> : null}
      </label>
      {field.kind === "textarea" || field.kind === "string-list" ? (
        <textarea
          id={id}
          name={field.name}
          required={field.required}
          rows={field.rows ?? (field.kind === "string-list" ? 5 : 4)}
          defaultValue={defaultValue}
          placeholder={field.placeholder}
          aria-describedby={describedBy}
          aria-invalid={fieldErrors?.length ? true : undefined}
          className={cn(controlStyles, "mt-2 min-h-28 resize-y", errorStyles)}
        />
      ) : field.kind === "select" || field.kind === "relation" ? (
        <select
          id={id}
          name={field.name}
          required={field.required}
          defaultValue={defaultValue ?? ""}
          aria-describedby={describedBy}
          aria-invalid={fieldErrors?.length ? true : undefined}
          className={cn(controlStyles, "mt-2", errorStyles)}
        >
          <option value="">{field.required ? "Select an option" : "Not specified"}</option>
          {options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          name={field.name}
          required={field.required}
          type={
            field.kind === "email"
              ? "email"
              : field.kind === "number"
                ? "number"
                : field.kind === "url"
                  ? "url"
                  : "text"
          }
          inputMode={field.kind === "number" ? "decimal" : undefined}
          min={field.min}
          max={field.max}
          step={field.step}
          pattern={field.kind === "slug" ? "[a-z0-9]+(?:-[a-z0-9]+)*" : undefined}
          autoComplete={field.autocomplete}
          defaultValue={defaultValue}
          placeholder={field.placeholder}
          aria-describedby={describedBy}
          aria-invalid={fieldErrors?.length ? true : undefined}
          className={cn(controlStyles, "mt-2", errorStyles)}
        />
      )}
      {field.description ? (
        <p id={descriptionId} className="mt-2 text-xs leading-5 text-muted">
          {field.description}
        </p>
      ) : field.kind === "string-list" ? (
        <p id={descriptionId} className="mt-2 text-xs leading-5 text-muted">
          Enter one item per line.
        </p>
      ) : null}
      {fieldErrors?.length ? (
        <p id={errorId} className="mt-2 text-xs leading-5 text-danger" role="alert">
          {fieldErrors.join(" ")}
        </p>
      ) : null}
    </div>
  );
}

export function AdminResourceForm({
  mode,
  record,
  relationOptions = {},
  resource,
}: AdminResourceFormProps) {
  const router = useRouter();
  const formRef = useRef<HTMLFormElement>(null);
  const sections = Array.from(new Set(resource.fields.map((item) => item.section)));
  const missingRequiredRelation = resource.fields.find(
    (field) =>
      field.kind === "relation" &&
      field.required &&
      field.optionsResource &&
      (relationOptions[field.optionsResource]?.length ?? 0) === 0,
  );

  const [state, formAction, pending] = useActionState(
    async (
      _previousState: AdminActionResult,
      formData: FormData,
    ): Promise<AdminActionResult> => {
      const input = serializeForm(resource.fields, formData);

      return mode === "create"
        ? createAdminRecordAction(resource.key, input)
        : updateAdminRecordAction(resource.key, record?.id ?? "", input);
    },
    initialActionState,
  );

  useEffect(() => {
    if (!state.ok) return;

    if (mode === "create" && state.record?.id) {
      router.replace(`/admin/${resource.key}/${state.record.id}/edit`);
    }
    router.refresh();
  }, [mode, resource.key, router, state.ok, state.record?.id]);

  useEffect(() => {
    if (state.ok || !state.fieldErrors) return;

    formRef.current
      ?.querySelector<HTMLElement>('[aria-invalid="true"]')
      ?.focus();
  }, [state.fieldErrors, state.ok]);

  return (
    <form
      ref={formRef}
      action={formAction}
      className="space-y-7"
      noValidate={false}
    >
      <div className="flex gap-3 rounded-2xl border border-line bg-ink/[0.025] p-4 text-sm leading-6 text-muted">
        <AlertTriangle aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-primary" />
        <p>
          Enter only information supported by the Academic CV or information you have independently verified. Blank fields are preferable to inferred details.
        </p>
      </div>

      {record?.status === "ARCHIVED" ? (
        <div className="rounded-2xl border border-secondary/30 bg-secondary-soft p-4 text-sm leading-6 text-ink">
          <p className="font-semibold">This record is archived</p>
          <p className="mt-1 text-muted">
            Choose Draft or Published under Visibility and save to restore it.
          </p>
        </div>
      ) : null}

      {missingRequiredRelation?.optionsResource ? (
        <div className="rounded-2xl border border-danger/30 bg-danger/[0.055] p-4 text-sm leading-6 text-ink">
          <p className="font-semibold">A related record is required</p>
          <p className="mt-1 text-muted">
            Create a {missingRequiredRelation.label.toLowerCase()} before saving this record.
          </p>
          <div className="mt-3">
            <ButtonLink
              href={`/admin/${missingRequiredRelation.optionsResource}/new`}
              size="sm"
              variant="outline"
            >
              Create {missingRequiredRelation.label.toLowerCase()}
            </ButtonLink>
          </div>
        </div>
      ) : null}

      {sections.map((section) => (
        <fieldset
          key={section}
          className="rounded-2xl border border-line bg-surface p-5 sm:p-6"
        >
          <legend className="px-1 font-serif text-xl font-medium tracking-[-0.02em] text-ink">
            {section}
          </legend>
          <div className="mt-2 grid gap-x-5 gap-y-6 sm:grid-cols-2">
            {resource.fields
              .filter((item) => item.section === section)
              .map((field) => (
                <AdminFormField
                  key={field.name}
                  field={field}
                  fieldErrors={state.fieldErrors?.[field.name]}
                  record={record}
                  relationOptions={relationOptions}
                />
              ))}
          </div>
        </fieldset>
      ))}

      {state.message ? (
        <div
          aria-live="polite"
          role={state.ok ? "status" : "alert"}
          className={cn(
            "rounded-xl border px-4 py-3 text-sm font-medium",
            state.ok
              ? "border-secondary/30 bg-secondary-soft text-secondary"
              : "border-danger/30 bg-danger/[0.06] text-danger",
          )}
        >
          {state.message}
        </div>
      ) : null}

      <div className="sticky bottom-4 z-10 flex flex-col-reverse gap-3 rounded-2xl border border-line bg-canvas/92 p-3 shadow-[0_18px_55px_-30px_rgb(20_25_35/0.4)] backdrop-blur-xl sm:flex-row sm:items-center sm:justify-end">
        <ButtonLink href={`/admin/${resource.key}`} variant="ghost">
          Cancel
        </ButtonLink>
        <Button
          type="submit"
          disabled={pending || Boolean(missingRequiredRelation)}
        >
          <Save aria-hidden="true" className="size-4" />
          {pending
            ? "Saving…"
            : mode === "create"
              ? `Create ${resource.singularLabel}`
              : "Save changes"}
        </Button>
      </div>
    </form>
  );
}
