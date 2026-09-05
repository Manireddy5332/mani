"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

function subscribeToHydration() {
  return () => undefined;
}

function getClientSnapshot() {
  return true;
}

function getServerSnapshot() {
  return false;
}

export function ThemeToggle() {
  const mounted = useSyncExternalStore(
    subscribeToHydration,
    getClientSnapshot,
    getServerSnapshot,
  );
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const label = isDark ? "Switch to light theme" : "Switch to dark theme";

  return (
    <button
      type="button"
      className="inline-flex size-10 shrink-0 items-center justify-center rounded-full border border-line bg-canvas text-ink transition-[color,background-color,border-color,transform] duration-200 hover:-translate-y-0.5 hover:border-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary active:translate-y-0 disabled:cursor-wait motion-reduce:transform-none motion-reduce:transition-none"
      aria-label={mounted ? label : "Theme selector loading"}
      aria-pressed={mounted ? isDark : undefined}
      disabled={!mounted}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {mounted && isDark ? (
        <Sun aria-hidden="true" className="size-[1.125rem]" strokeWidth={1.8} />
      ) : (
        <Moon aria-hidden="true" className="size-[1.125rem]" strokeWidth={1.8} />
      )}
    </button>
  );
}
