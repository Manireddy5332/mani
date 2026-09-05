import type { HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type SurfaceElement = "article" | "aside" | "div" | "section";
export type SurfaceVariant = "outlined" | "raised" | "subtle" | "accent";
export type SurfacePadding = "none" | "sm" | "md" | "lg";

export interface SurfaceProps extends HTMLAttributes<HTMLElement> {
  as?: SurfaceElement;
  padding?: SurfacePadding;
  variant?: SurfaceVariant;
}

const variantStyles: Record<SurfaceVariant, string> = {
  outlined: "border-line bg-canvas",
  raised:
    "border-line/80 bg-canvas shadow-[0_24px_80px_-48px_rgb(20_25_35/0.45)]",
  subtle: "border-line/70 bg-ink/[0.025]",
  accent: "border-primary/25 bg-primary/[0.055]",
};

const paddingStyles: Record<SurfacePadding, string> = {
  none: "p-0",
  sm: "p-4 sm:p-5",
  md: "p-5 sm:p-7",
  lg: "p-6 sm:p-8 lg:p-10",
};

export function Surface({
  as: Component = "div",
  className,
  padding = "md",
  variant = "outlined",
  ...props
}: SurfaceProps) {
  return (
    <Component
      className={cn(
        "rounded-2xl border text-ink",
        variantStyles[variant],
        paddingStyles[padding],
        className,
      )}
      {...props}
    />
  );
}
