import { AlertCircle, ArrowRight, Database, FileCheck2 } from "lucide-react";
import Link from "next/link";

import { AdminIcon } from "@/components/admin/admin-icon";
import { AdminPageHeader } from "@/components/admin/admin-page-header";
import {
  adminNavigationGroups,
  getAdminResource,
  getAdminResourcePath,
} from "@/features/admin/registry";
import type { AdminDashboardDto } from "@/features/admin/types";

type AdminDashboardProps = {
  dashboard: AdminDashboardDto;
};

export function AdminDashboard({ dashboard }: AdminDashboardProps) {
  return (
    <div>
      <AdminPageHeader
        eyebrow="Content overview"
        title="A clear view of the portfolio."
        description="Manage private drafts and verified portfolio records without changing source files. Published saves revalidate the connected public portfolio, while drafts remain private."
      />

      {!dashboard.profileExists ? (
        <div className="mt-8 flex gap-4 rounded-2xl border border-primary/25 bg-primary/[0.055] p-5 text-sm leading-6 text-ink">
          <AlertCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-primary" />
          <div>
            <p className="font-semibold">Create the primary profile first</p>
            <p className="mt-1 text-muted">
              Profile-linked collections remain unavailable until a primary profile exists. No content is generated automatically.
            </p>
            <Link
              href="/admin/profile/new"
              className="mt-3 inline-flex items-center gap-1.5 font-semibold text-primary hover:underline"
            >
              Create profile
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          </div>
        </div>
      ) : null}

      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <div className="rounded-2xl border border-line bg-surface p-5">
          <span className="grid size-10 place-items-center rounded-xl bg-ink/[0.055] text-ink">
            <Database aria-hidden="true" className="size-5" />
          </span>
          <p className="mt-5 font-mono text-[0.68rem] font-semibold tracking-[0.14em] text-muted uppercase">
            Managed records
          </p>
          <p className="mt-1 font-serif text-4xl font-medium tracking-[-0.04em] text-ink">
            {dashboard.totalRecords}
          </p>
        </div>
        <div className="rounded-2xl border border-line bg-surface p-5 sm:col-span-1 xl:col-span-2">
          <span className="grid size-10 place-items-center rounded-xl bg-secondary-soft text-secondary">
            <FileCheck2 aria-hidden="true" className="size-5" />
          </span>
          <p className="mt-5 text-sm font-semibold text-ink">Content integrity remains explicit</p>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
            Empty collections are intentional. Add or publish only information supported by the Academic CV or information you have separately verified.
          </p>
        </div>
      </div>

      <div className="mt-10 space-y-10">
        {adminNavigationGroups.map((group) => {
          const stats = dashboard.resources
            .map((stat) => ({ stat, resource: getAdminResource(stat.resource) }))
            .filter(
              (entry) => entry.resource?.navigationGroup === group,
            );

          if (stats.length === 0) return null;

          return (
            <section key={group} aria-labelledby={`dashboard-${group.toLowerCase()}`}>
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="font-mono text-[0.68rem] font-semibold tracking-[0.14em] text-primary uppercase">
                    {group}
                  </p>
                  <h2
                    id={`dashboard-${group.toLowerCase()}`}
                    className="mt-2 font-serif text-2xl font-medium tracking-[-0.025em] text-ink"
                  >
                    Manage {group.toLowerCase()}
                  </h2>
                </div>
              </div>
              <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {stats.map(({ resource, stat }) => {
                  if (!resource) return null;

                  return (
                    <Link
                      key={resource.key}
                      href={getAdminResourcePath(resource.key)}
                      className="group rounded-2xl border border-line bg-surface p-5 transition-[border-color,background-color,transform] hover:-translate-y-0.5 hover:border-primary/45 hover:bg-surface-strong focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary motion-reduce:transform-none motion-reduce:transition-none"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <span className="grid size-10 place-items-center rounded-xl bg-primary/10 text-primary">
                          <AdminIcon className="size-5" name={resource.icon} />
                        </span>
                        <ArrowRight
                          aria-hidden="true"
                          className="size-4 text-muted transition-transform group-hover:translate-x-0.5 motion-reduce:transform-none"
                        />
                      </div>
                      <h3 className="mt-5 font-semibold text-ink">{resource.label}</h3>
                      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
                        <span>{stat.count} total</span>
                        <span>{stat.draftCount} draft</span>
                        <span>{stat.publishedCount} published</span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
