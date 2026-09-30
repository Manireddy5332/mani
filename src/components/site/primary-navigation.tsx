"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { MobileNavigation } from "@/components/ui/mobile-navigation";
import { cn } from "@/lib/cn";

export const primaryNavigation = [
  { label: "Home", href: "/" },
  { label: "Research", href: "/research" },
  { label: "Projects", href: "/projects" },
  { label: "Experience", href: "/experience" },
  { label: "Writing", href: "/writing" },
  { label: "About", href: "/about" },
] as const;

export function PrimaryNavigation() {
  const pathname = usePathname();

  const links = primaryNavigation.map((item) => {
    const isCurrent =
      item.href === "/"
        ? pathname === "/"
        : pathname === item.href || pathname.startsWith(`${item.href}/`);

    return (
      <Link
        key={item.href}
        href={item.href}
        aria-current={isCurrent ? "page" : undefined}
        className={cn(
          "inline-flex min-h-11 items-center justify-center rounded-full border px-3 py-2 font-mono text-xs font-medium uppercase tracking-[0.12em] transition-[color,background-color,border-color] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary motion-reduce:transition-none lg:shrink-0 lg:text-[0.68rem]",
          isCurrent
            ? "border-ink bg-ink text-canvas hover:border-primary hover:bg-primary hover:text-primary-contrast"
            : "border-transparent text-muted hover:border-line hover:bg-surface hover:text-ink",
        )}
      >
        {item.label}
      </Link>
    );
  });

  return (
    <div className="min-w-0 border-t border-line/70 py-1 lg:shrink-0 lg:border-t-0 lg:py-0">
      <MobileNavigation label="Menu" routeKey={pathname}>
        <nav aria-label="Primary navigation" className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {links}
        </nav>
      </MobileNavigation>
      <nav aria-label="Primary navigation" className="hidden items-center gap-1 lg:flex">
        {links}
      </nav>
    </div>
  );
}
