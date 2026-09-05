"use client";

import { ArrowDown, ArrowUp, Plus, Save, Trash2 } from "lucide-react";

import { Button } from "@/components/ui";
import { cn } from "@/lib/cn";

export const nestedControlStyles =
  "min-h-11 w-full rounded-xl border border-line bg-canvas px-3.5 py-2.5 text-sm text-ink shadow-[0_1px_0_rgb(20_25_35/0.02)] transition-[border-color,box-shadow] placeholder:text-muted/65 focus:border-primary focus:outline-none focus:ring-3 focus:ring-primary/10 disabled:cursor-not-allowed disabled:opacity-55";

type NestedTextFieldProps = {
  description?: string;
  errors?: string[];
  id: string;
  label: string;
  multiline?: boolean;
  onChange: (value: string) => void;
  placeholder?: string;
  required?: boolean;
  rows?: number;
  slug?: boolean;
  value: string;
};

export function NestedTextField({
  description,
  errors,
  id,
  label,
  multiline = false,
  onChange,
  placeholder,
  required = false,
  rows = 4,
  slug = false,
  value,
}: NestedTextFieldProps) {
  const descriptionId = description ? `${id}-description` : undefined;
  const errorId = errors?.length ? `${id}-error` : undefined;
  const describedBy = [descriptionId, errorId].filter(Boolean).join(" ") || undefined;
  const styles = cn(
    nestedControlStyles,
    "mt-2",
    multiline && "min-h-28 resize-y",
    errors?.length && "border-danger focus:border-danger focus:ring-danger/10",
  );
  const commonProps = {
    id,
    name: id,
    value,
    required,
    placeholder,
    "aria-describedby": describedBy,
    "aria-invalid": errors?.length ? (true as const) : undefined,
    className: styles,
  };

  return (
    <div>
      <label htmlFor={id} className="block text-sm font-semibold text-ink">
        {label}
        {required ? (
          <span className="ml-1 text-danger" aria-hidden="true">
            *
          </span>
        ) : null}
      </label>
      {multiline ? (
        <textarea
          {...commonProps}
          rows={rows}
          onChange={(event) => onChange(event.target.value)}
        />
      ) : (
        <input
          {...commonProps}
          type="text"
          pattern={slug ? "[a-z0-9]+(?:-[a-z0-9]+)*" : undefined}
          autoComplete="off"
          onChange={(event) => onChange(event.target.value)}
        />
      )}
      {description ? (
        <p id={descriptionId} className="mt-2 text-xs leading-5 text-muted">
          {description}
        </p>
      ) : null}
      {errors?.length ? (
        <p id={errorId} className="mt-2 text-xs leading-5 text-danger" role="alert">
          {errors.join(" ")}
        </p>
      ) : null}
    </div>
  );
}

type NestedItemHeaderProps = {
  index: number;
  itemLabel: string;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
  pending: boolean;
  title: string;
  total: number;
};

export function NestedItemHeader({
  index,
  itemLabel,
  onMove,
  onRemove,
  pending,
  title,
  total,
}: NestedItemHeaderProps) {
  return (
    <div className="flex flex-col gap-3 border-b border-line pb-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <p className="font-mono text-[0.65rem] font-semibold tracking-[0.13em] text-muted uppercase">
          {itemLabel} {index + 1} of {total}
        </p>
        <h4 className="mt-1 truncate text-sm font-semibold text-ink">{title}</h4>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-1">
        <Button
          type="button"
          size="icon"
          variant="ghost"
          disabled={pending || index === 0}
          onClick={() => onMove(-1)}
          aria-label={`Move ${itemLabel.toLowerCase()} ${index + 1} up`}
          title="Move up"
        >
          <ArrowUp aria-hidden="true" className="size-4" />
        </Button>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          disabled={pending || index === total - 1}
          onClick={() => onMove(1)}
          aria-label={`Move ${itemLabel.toLowerCase()} ${index + 1} down`}
          title="Move down"
        >
          <ArrowDown aria-hidden="true" className="size-4" />
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          disabled={pending}
          onClick={onRemove}
          className="text-danger hover:text-danger"
          aria-label={`Remove ${title || `${itemLabel.toLowerCase()} ${index + 1}`}`}
        >
          <Trash2 aria-hidden="true" className="size-4" />
          Remove
        </Button>
      </div>
    </div>
  );
}

type NestedCollectionHeaderProps = {
  addButtonId: string;
  addLabel: string;
  description: string;
  onAdd: () => void;
  pending: boolean;
  title: string;
};

export function NestedCollectionHeader({
  addButtonId,
  addLabel,
  description,
  onAdd,
  pending,
  title,
}: NestedCollectionHeaderProps) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="max-w-2xl">
        <h3 className="font-serif text-2xl font-medium tracking-[-0.025em] text-ink">
          {title}
        </h3>
        <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
      </div>
      <Button
        id={addButtonId}
        type="button"
        size="sm"
        variant="outline"
        disabled={pending}
        onClick={onAdd}
      >
        <Plus aria-hidden="true" className="size-4" />
        {addLabel}
      </Button>
    </div>
  );
}

export function focusNestedAddButton(id: string) {
  requestAnimationFrame(() => document.getElementById(id)?.focus());
}

type NestedSaveBarProps = {
  dirty: boolean;
  pending: boolean;
  removedCount: number;
};

export function NestedSaveBar({
  dirty,
  pending,
  removedCount,
}: NestedSaveBarProps) {
  return (
    <div className="sticky bottom-4 z-10 flex flex-col gap-3 rounded-2xl border border-line bg-canvas/92 p-3 shadow-[0_18px_55px_-30px_rgb(20_25_35/0.4)] backdrop-blur-xl sm:flex-row sm:items-center sm:justify-between">
      <p className="px-1 text-xs leading-5 text-muted">
        {removedCount > 0
          ? `${removedCount} saved ${removedCount === 1 ? "item" : "items"} will require confirmation before removal.`
          : dirty
            ? "Unsaved nested-content changes."
            : "Nested content is up to date."}
      </p>
      <Button type="submit" disabled={pending || !dirty}>
        <Save aria-hidden="true" className="size-4" />
        {pending ? "Saving…" : "Save nested content"}
      </Button>
    </div>
  );
}

export function getNestedFieldErrors(
  fieldErrors: Record<string, string[]> | undefined,
  path: string,
): string[] | undefined {
  if (!fieldErrors) return undefined;

  const bracketPath = path.replace(/\.(\d+)\./g, "[$1].");
  const messages = [
    ...(fieldErrors[path] ?? []),
    ...(fieldErrors[bracketPath] ?? []),
  ];

  return messages.length ? Array.from(new Set(messages)) : undefined;
}

export function getNestedSummaryErrors(
  fieldErrors: Record<string, string[]> | undefined,
): string[] {
  if (!fieldErrors) return [];

  return Array.from(
    new Set(
      Object.entries(fieldErrors)
        .filter(([path]) => !/\.\d+\./.test(path))
        .flatMap(([, messages]) => messages),
    ),
  );
}

export function confirmNestedRemoval(labels: string[]): boolean {
  const visibleLabels = labels.slice(0, 8);
  const remaining = labels.length - visibleLabels.length;
  const list = visibleLabels.map((label) => `• ${label}`).join("\n");
  const more = remaining > 0 ? `\n• and ${remaining} more` : "";

  return window.confirm(
    `Remove the following nested content? This cannot be undone after saving.\n\n${list}${more}`,
  );
}

export function moveNestedItem<T>(
  items: readonly T[],
  index: number,
  direction: -1 | 1,
): T[] {
  const nextIndex = index + direction;
  if (nextIndex < 0 || nextIndex >= items.length) return [...items];

  const next = [...items];
  [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
  return next;
}
