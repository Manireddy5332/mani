import type { ComponentPropsWithoutRef, ReactNode } from "react";

import { cn } from "@/lib/cn";

type HeadingElement = "h1" | "h2" | "h3";
export type SectionHeadingAlign = "start" | "center";
export type SectionHeadingSize = "sm" | "md" | "lg";

export interface SectionHeadingProps
  extends Omit<ComponentPropsWithoutRef<"header">, "title"> {
  actions?: ReactNode;
  align?: SectionHeadingAlign;
  description?: ReactNode;
  eyebrow?: ReactNode;
  size?: SectionHeadingSize;
  title: ReactNode;
  titleAs?: HeadingElement;
}

const titleSizeStyles: Record<SectionHeadingSize, string> = {
  sm: "text-2xl sm:text-3xl",
  md: "text-3xl sm:text-4xl lg:text-5xl",
  lg: "text-4xl sm:text-5xl lg:text-6xl",
};

export function SectionHeading({
  actions,
  align = "start",
  className,
  description,
  eyebrow,
  size = "md",
  title,
  titleAs: Title = "h2",
  ...props
}: SectionHeadingProps) {
  const centered = align === "center";

  return (
    <header
      className={cn(
        "flex w-full flex-col gap-5 sm:flex-row sm:items-end sm:justify-between",
        centered && "items-center text-center sm:flex-col sm:items-center",
        className,
      )}
      {...props}
    >
      <div
        className={cn(
          "flex max-w-3xl flex-col gap-3",
          centered && "items-center",
        )}
      >
        {eyebrow ? (
          <p className="font-mono text-xs font-semibold tracking-[0.16em] text-primary uppercase">
            {eyebrow}
          </p>
        ) : null}
        <Title
          className={cn(
            "text-balance font-serif leading-[1.05] font-medium tracking-[-0.035em] text-ink",
            titleSizeStyles[size],
          )}
        >
          {title}
        </Title>
        {description ? (
          <div className="max-w-2xl text-base leading-7 text-ink/70 sm:text-lg sm:leading-8">
            {description}
          </div>
        ) : null}
      </div>
      {actions ? (
        <div
          className={cn(
            "flex shrink-0 flex-wrap gap-3",
            centered && "justify-center",
          )}
        >
          {actions}
        </div>
      ) : null}
    </header>
  );
}
