"use client";

import { LayoutDashboard } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { AdminIcon } from "@/components/admin/admin-icon";
import {
  adminNavigationGroups,
  adminResources,
  getAdminResourcePath,
} from "@/features/admin/registry";
import { cn } from "@/lib/cn";

const linkStyles =
  "group flex min-h-10 items-center gap-3 rounded-xl border px-3 py-2 text-sm font-semibold transition-[color,background-color,border-color] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary motion-reduce:transition-none";

export function AdminNavigation() {
  const pathname = usePathname();

  return (
    <nav aria-label="Admin navigation" className="min-w-0">
      <Link
        href="/admin"
        prefetch={false}
        aria-current={pathname === "/admin" ? "page" : undefined}
        className={cn(
          linkStyles,
          pathname === "/admin"
            ? "border-primary/25 bg-primary/10 text-primary"
            : "border-transparent text-ink/68 hover:border-line hover:bg-ink/[0.035] hover:text-ink",
        )}
      >
        <LayoutDashboard aria-hidden="true" className="size-4 shrink-0" />
        Overview
      </Link>

      <div className="mt-5 space-y-6">
        {adminNavigationGroups.map((group) => {
          const resources = adminResources.filter(
            (resource) => resource.navigationGroup === group,
          );

          return (
            <div key={group}>
              <p className="px-3 font-mono text-[0.64rem] font-semibold tracking-[0.16em] text-muted uppercase">
                {group}
              </p>
              <div className="mt-2 space-y-1">
                {resources.map((resource) => {
                  const href = getAdminResourcePath(resource.key);
                  const current =
                    pathname === href || pathname.startsWith(`${href}/`);

                  return (
                    <Link
                      key={resource.key}
                      href={href}
                      prefetch={false}
                      aria-current={current ? "page" : undefined}
                      className={cn(
                        linkStyles,
                        current
                          ? "border-primary/25 bg-primary/10 text-primary"
                          : "border-transparent text-ink/68 hover:border-line hover:bg-ink/[0.035] hover:text-ink",
                      )}
                    >
                      <AdminIcon
                        className="size-4 shrink-0"
                        name={resource.icon}
                      />
                      <span className="truncate">{resource.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </nav>
  );
}
