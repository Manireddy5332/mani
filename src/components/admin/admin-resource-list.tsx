import { Pencil, Plus } from "lucide-react";

import { AdminDeleteRecordButton } from "@/components/admin/admin-delete-record-button";
import { AdminEmptyState } from "@/components/admin/admin-empty-state";
import { AdminPageHeader } from "@/components/admin/admin-page-header";
import { AdminRecordVisibilityButton } from "@/components/admin/admin-record-visibility-button";
import { AdminStatusBadge } from "@/components/admin/admin-status-badge";
import { ButtonLink } from "@/components/ui";
import type {
  AdminRecordSummaryDto,
  AdminResourceDefinition,
} from "@/features/admin/types";

type AdminResourceListProps = {
  profileExists: boolean;
  records: AdminRecordSummaryDto[];
  resource: AdminResourceDefinition;
};

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeZone: "UTC",
});

function displayDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : dateFormatter.format(date);
}

export function AdminResourceList({
  profileExists,
  records,
  resource,
}: AdminResourceListProps) {
  const isProfile = resource.key === "profile";
  const profileBlocksCreation =
    !profileExists && !isProfile && resource.key !== "site-settings";
  const singletonExists = isProfile && records.length > 0;
  const canCreate =
    resource.capabilities.create && !profileBlocksCreation && !singletonExists;
  const newHref = `/admin/${resource.key}/new`;

  return (
    <div>
      <AdminPageHeader
        eyebrow={resource.navigationGroup}
        title={resource.label}
        description={resource.description}
        actions={
          canCreate ? (
            <ButtonLink href={newHref} size="sm">
              <Plus aria-hidden="true" className="size-4" />
              New {resource.singularLabel}
            </ButtonLink>
          ) : undefined
        }
      />

      <div className="mt-8">
        {records.length === 0 ? (
          <AdminEmptyState
            title={
              profileBlocksCreation
                ? "Create the primary profile first"
                : resource.emptyTitle
            }
            description={
              profileBlocksCreation
                ? "This collection belongs to the primary profile. Create that profile before adding related records; no placeholder content will be generated."
                : resource.emptyDescription
            }
            actionHref={
              profileBlocksCreation
                ? "/admin/profile/new"
                : canCreate
                  ? newHref
                  : undefined
            }
            actionLabel={
              profileBlocksCreation
                ? "Create profile"
                : canCreate
                  ? `Create ${resource.singularLabel}`
                  : undefined
            }
          />
        ) : (
          <div className="overflow-hidden rounded-2xl border border-line bg-surface">
            <div
              aria-label={`${resource.label} table; scroll horizontally for all columns`}
              className="overflow-x-auto focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              role="region"
              tabIndex={0}
            >
              <table className="w-full min-w-[46rem] border-collapse text-left">
                <caption className="sr-only">
                  {resource.label} records
                </caption>
                <thead className="border-b border-line bg-ink/[0.025]">
                  <tr className="font-mono text-[0.67rem] font-semibold tracking-[0.12em] text-muted uppercase">
                    <th scope="col" className="px-5 py-4">Record</th>
                    <th scope="col" className="px-5 py-4">Visibility</th>
                    <th scope="col" className="px-5 py-4">Order</th>
                    <th scope="col" className="px-5 py-4">Updated</th>
                    <th scope="col" className="px-5 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {records.map((record) => (
                    <tr key={record.id} className="align-top">
                      <th scope="row" className="max-w-md px-5 py-5 font-normal">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-semibold text-ink">{record.title}</span>
                          {record.featured ? (
                            <span className="rounded-full border border-secondary/30 bg-secondary-soft px-2 py-0.5 font-mono text-[0.62rem] font-semibold tracking-[0.08em] text-secondary uppercase">
                              Featured
                            </span>
                          ) : null}
                        </div>
                        {record.subtitle ? (
                          <p className="mt-1 line-clamp-2 text-sm leading-6 text-muted">
                            {record.subtitle}
                          </p>
                        ) : null}
                      </th>
                      <td className="px-5 py-5"><AdminStatusBadge status={record.status} /></td>
                      <td className="px-5 py-5 text-sm text-muted">
                        {record.sortOrder ?? "—"}
                      </td>
                      <td className="px-5 py-5 text-sm text-muted">
                        <time dateTime={record.updatedAt}>{displayDate(record.updatedAt)}</time>
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex items-start justify-end gap-1">
                          {resource.key !== "profile" &&
                          (record.status === "DRAFT" ||
                            record.status === "PUBLISHED") ? (
                            <AdminRecordVisibilityButton
                              id={record.id}
                              resource={resource.key}
                              status={record.status}
                              title={record.title}
                            />
                          ) : null}
                          {resource.capabilities.update ? (
                            <ButtonLink
                              href={`/admin/${resource.key}/${record.id}/edit`}
                              size="sm"
                              variant="ghost"
                            >
                              <Pencil aria-hidden="true" className="size-4" />
                              Edit
                            </ButtonLink>
                          ) : null}
                          {resource.capabilities.delete && record.status !== "ARCHIVED" ? (
                            <AdminDeleteRecordButton
                              id={record.id}
                              resource={resource.key}
                              title={record.title}
                            />
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
