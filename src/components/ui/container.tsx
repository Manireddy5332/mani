import type { HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type ContainerElement = "div" | "header" | "main" | "section";
export type ContainerSize = "narrow" | "content" | "wide" | "fluid";

export interface ContainerProps extends HTMLAttributes<HTMLElement> {
  as?: ContainerElement;
  size?: ContainerSize;
}

const sizeStyles: Record<ContainerSize, string> = {
  narrow: "max-w-3xl",
  content: "max-w-6xl",
  wide: "max-w-[90rem]",
  fluid: "max-w-none",
};

export function Container({
  as: Component = "div",
  className,
  size = "wide",
  ...props
}: ContainerProps) {
  return (
    <Component
      className={cn(
        "mx-auto w-full px-5 sm:px-8 lg:px-12",
        sizeStyles[size],
        className,
      )}
      {...props}
    />
  );
}
