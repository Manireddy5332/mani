"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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

  return (
    <nav
      aria-label="Primary navigation"
      className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto border-t border-line/70 py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden lg:justify-end lg:overflow-visible lg:border-t-0 lg:py-0"
    >
      {primaryNavigation.map((item) => {
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
              "shrink-0 rounded-full border px-3 py-1.5 font-mono text-[0.68rem] font-medium uppercase tracking-[0.12em] transition-[color,background-color,border-color] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary motion-reduce:transition-none",
              isCurrent
                ? "border-ink bg-ink text-canvas hover:border-primary hover:bg-primary hover:text-primary-contrast"
                : "border-transparent text-muted hover:border-line hover:bg-surface hover:text-ink",
            )}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
