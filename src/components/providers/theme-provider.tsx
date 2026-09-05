"use client";

import type { ComponentProps } from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

type ThemeProviderProps = ComponentProps<typeof NextThemesProvider>;

/**
 * Keeps theme configuration in one client boundary while allowing the rest of
 * the application shell to remain server-rendered.
 */
export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      storageKey="manikanta-portfolio-theme"
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
