import type { ComponentPropsWithoutRef } from "react";

import { cn } from "@/lib/cn";

export type BadgeVariant = "neutral" | "accent" | "outline";

export interface BadgeProps extends ComponentPropsWithoutRef<"span"> {
  variant?: BadgeVariant;
}

const variantStyles: Record<BadgeVariant, string> = {
  neutral: "border-line bg-ink/[0.045] text-ink",
  accent: "border-primary/25 bg-primary/10 text-primary",
  outline: "border-line bg-transparent text-ink/75",
};

export function Badge({
  className,
  variant = "neutral",
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 items-center rounded-full border px-2.5 py-1 font-mono text-[0.6875rem] font-semibold tracking-[0.08em] uppercase",
        variantStyles[variant],
        className,
      )}
      {...props}
    />
  );
}
