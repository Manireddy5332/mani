"use client";

import type { HTMLMotionProps } from "motion/react";
import { motion, useReducedMotion } from "motion/react";

export interface RevealProps
  extends Omit<
    HTMLMotionProps<"div">,
    "animate" | "initial" | "transition" | "viewport" | "whileInView"
  > {
  amount?: "all" | "some" | number;
  delay?: number;
  distance?: number;
  once?: boolean;
}

const revealEase = [0.22, 1, 0.36, 1] as const;

export function Reveal({
  amount = 0.18,
  children,
  delay = 0,
  distance = 20,
  once = true,
  ...props
}: RevealProps) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={
        shouldReduceMotion ? false : { opacity: 0, translateY: distance }
      }
      transition={
        shouldReduceMotion
          ? { duration: 0 }
          : { delay, duration: 0.52, ease: revealEase }
      }
      viewport={{ amount, once }}
      whileInView={{ opacity: 1, translateY: 0 }}
      {...props}
    >
      {children}
    </motion.div>
  );
}
