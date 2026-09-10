import { ShieldCheck } from "lucide-react";

import { SignOutButton } from "@/components/admin/auth-buttons";
import { AdminNavigation } from "@/components/admin/admin-navigation";
import { siteConfig } from "@/lib/site";

type AdminShellProps = {
  children: React.ReactNode;
  user: {
    email: string;
    name?: string | null;
  };
};

function getInitials(name: string | null | undefined, email: string) {
  const source = name?.trim() || email.split("@")[0] || "A";
  const parts = source.split(/\s+/).filter(Boolean);

  return parts
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

export function AdminShell({ children, user }: AdminShellProps) {
  return (
    <main id="main-content" className="flex-1">
      <section className="border-b border-line bg-surface/60">
        <div className="mx-auto flex w-full max-w-[90rem] flex-col gap-5 px-5 py-7 sm:px-8 md:flex-row md:items-center md:justify-between lg:px-12">
          <div>
            <div className="flex items-center gap-2 font-mono text-[0.7rem] font-semibold tracking-[0.16em] text-primary uppercase">
              <ShieldCheck aria-hidden="true" className="size-4" />
              Private administration
            </div>
            <p className="mt-2 font-serif text-3xl leading-tight font-medium tracking-[-0.035em] text-ink sm:text-4xl">
              {siteConfig.adminName}
            </p>
          </div>

          <div className="flex min-w-0 items-center gap-3 rounded-2xl border border-line bg-canvas p-2 pr-3">
            <span
              aria-hidden="true"
              className="grid size-10 shrink-0 place-items-center rounded-xl bg-ink font-mono text-xs font-semibold tracking-[0.08em] text-canvas"
            >
              {getInitials(user.name, user.email)}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold text-ink">
                {user.name || "Administrator"}
              </span>
              <span className="block truncate text-xs text-muted">
                {user.email}
              </span>
            </span>
            <SignOutButton />
          </div>
        </div>
      </section>

      <div className="mx-auto grid w-full max-w-[90rem] gap-8 px-5 py-8 sm:px-8 lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-10 lg:px-12 lg:py-12">
        <aside className="min-w-0 lg:sticky lg:top-28 lg:self-start">
          <div className="max-h-[calc(100vh-9rem)] overflow-y-auto rounded-2xl border border-line bg-surface/70 p-3 shadow-[0_22px_60px_-48px_rgb(20_25_35/0.48)]">
            <AdminNavigation />
          </div>
        </aside>
        <div className="min-w-0">{children}</div>
      </div>
    </main>
  );
}
