"use client";

import { ChevronDown, Menu } from "lucide-react";
import type { ReactNode } from "react";

type MobileNavigationProps = {
  children: ReactNode;
  label: string;
  routeKey: string;
};

/** Native disclosure stays keyboard-accessible even before hydration. */
export function MobileNavigation({ children, label, routeKey }: MobileNavigationProps) {
  return (
    <details
      key={routeKey}
      className="group/mobile-nav lg:hidden"
      onKeyDown={(event) => {
        if (event.key !== "Escape") return;
        event.currentTarget.open = false;
        event.currentTarget.querySelector("summary")?.focus();
      }}
      onClick={(event) => {
        if ((event.target as HTMLElement).closest("a[href]")) {
          event.currentTarget.open = false;
        }
      }}
    >
      <summary className="flex min-h-11 cursor-pointer list-none items-center gap-3 rounded-xl px-3 py-2 text-sm font-semibold text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary [&::-webkit-details-marker]:hidden">
        <Menu aria-hidden="true" className="size-4 shrink-0" />
        <span className="min-w-0 flex-1">{label}</span>
        <ChevronDown aria-hidden="true" className="size-4 shrink-0 transition-transform group-open/mobile-nav:rotate-180 motion-reduce:transition-none" />
      </summary>
      <div className="max-h-[60svh] overflow-y-auto overscroll-contain px-1 py-3">
        {children}
      </div>
    </details>
  );
}
