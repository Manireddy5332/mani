import type { ComponentPropsWithoutRef } from "react";
import Link from "next/link";

import { cn } from "@/lib/cn";

export type ButtonVariant = "primary" | "secondary" | "outline" | "ghost";
export type ButtonSize = "sm" | "md" | "lg" | "icon";

export interface ButtonStyleProps {
  size?: ButtonSize;
  variant?: ButtonVariant;
  wide?: boolean;
}

const baseStyles =
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-full border text-sm font-semibold tracking-[-0.01em] transition-[color,background-color,border-color,box-shadow,transform] duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:pointer-events-none disabled:opacity-45 motion-safe:hover:-translate-y-px motion-reduce:transition-none";

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "border-primary bg-primary text-primary-contrast shadow-[0_12px_30px_-18px_currentColor] hover:bg-primary/90",
  secondary:
    "border-line bg-canvas text-ink hover:border-primary hover:text-primary",
  outline:
    "border-line bg-transparent text-ink hover:border-primary hover:text-primary",
  ghost:
    "border-transparent bg-transparent text-ink hover:bg-ink/[0.06] hover:text-primary",
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: "min-h-9 px-4 py-2 text-xs",
  md: "min-h-11 px-5 py-2.5",
  lg: "min-h-12 px-6 py-3 text-base",
  icon: "size-11 p-0",
};

export function buttonStyles({
  className,
  size = "md",
  variant = "primary",
  wide = false,
}: ButtonStyleProps & { className?: string } = {}) {
  return cn(
    baseStyles,
    variantStyles[variant],
    sizeStyles[size],
    wide && "w-full",
    className,
  );
}

export type ButtonProps = ComponentPropsWithoutRef<"button"> & ButtonStyleProps;

export function Button({
  className,
  size,
  type = "button",
  variant,
  wide,
  ...props
}: ButtonProps) {
  return (
    <button
      className={buttonStyles({ className, size, variant, wide })}
      type={type}
      {...props}
    />
  );
}

export type ButtonLinkProps = ComponentPropsWithoutRef<typeof Link> &
  ButtonStyleProps;

export function ButtonLink({
  className,
  size,
  variant,
  wide,
  ...props
}: ButtonLinkProps) {
  return (
    <Link
      className={buttonStyles({ className, size, variant, wide })}
      {...props}
    />
  );
}
