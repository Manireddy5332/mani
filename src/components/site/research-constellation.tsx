import { cn } from "@/lib/cn";

import styles from "./public-visuals.module.css";

export type ResearchConstellationProps = Readonly<{
  className?: string;
}>;

/**
 * A deterministic, decorative research network for the public homepage.
 * Fixed geometry keeps the server and client output identical while CSS owns
 * the optional motion and disables it when reduced motion is requested.
 */
export function ResearchConstellation({
  className,
}: ResearchConstellationProps) {
  return (
    <svg
      aria-hidden="true"
      className={cn(styles.constellation, className)}
      fill="none"
      focusable="false"
      viewBox="0 0 620 560"
    >
      <defs>
        <radialGradient id="constellation-halo" cx="0" cy="0" r="1">
          <stop stopColor="var(--ds-primary)" stopOpacity="0.24" />
          <stop offset="1" stopColor="var(--ds-primary)" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="constellation-line" x1="70" x2="548" y1="85" y2="485">
          <stop stopColor="var(--ds-primary)" stopOpacity="0.72" />
          <stop offset="0.52" stopColor="var(--ds-secondary)" stopOpacity="0.64" />
          <stop offset="1" stopColor="var(--ds-primary)" stopOpacity="0.2" />
        </linearGradient>
      </defs>

      <circle
        className={styles.ambientHalo}
        cx="330"
        cy="272"
        fill="url(#constellation-halo)"
        r="246"
      />
      <g className={styles.orbit} opacity="0.55" stroke="var(--ds-line-strong)">
        <ellipse cx="326" cy="277" rx="252" ry="171" strokeDasharray="3 10" />
        <ellipse
          cx="326"
          cy="277"
          rx="192"
          ry="254"
          strokeDasharray="2 13"
          transform="rotate(34 326 277)"
        />
        <circle cx="326" cy="277" r="112" strokeDasharray="1 9" />
      </g>

      <g stroke="url(#constellation-line)" strokeLinecap="round">
        <path className={styles.signalPath} d="M87 374 173 253 286 307 381 157 528 222" />
        <path className={styles.signalPathSlow} d="m125 119 148 72 108-34 91 198-168 91-131-193" />
        <path d="m87 374 217 72 224-224" opacity="0.28" />
        <path d="M125 119 87 374m441-152-56 133" opacity="0.28" />
      </g>

      <g fill="var(--ds-canvas)" stroke="var(--ds-primary)" strokeWidth="2">
        <g className={cn(styles.node, styles.nodeDelayOne)}>
          <circle cx="125" cy="119" r="8" />
          <circle cx="125" cy="119" fill="var(--ds-primary)" r="2.5" stroke="none" />
        </g>
        <g className={cn(styles.node, styles.nodeDelayTwo)}>
          <circle cx="173" cy="253" r="10" />
          <circle cx="173" cy="253" fill="var(--ds-primary)" r="3" stroke="none" />
        </g>
        <g className={styles.node}>
          <circle cx="286" cy="307" r="7" />
          <circle cx="286" cy="307" fill="var(--ds-secondary)" r="2.5" stroke="none" />
        </g>
        <g className={cn(styles.node, styles.nodeDelayThree)}>
          <circle cx="381" cy="157" r="11" />
          <circle cx="381" cy="157" fill="var(--ds-primary)" r="3.5" stroke="none" />
        </g>
        <g className={cn(styles.node, styles.nodeDelayOne)}>
          <circle cx="528" cy="222" r="8" />
          <circle cx="528" cy="222" fill="var(--ds-secondary)" r="2.5" stroke="none" />
        </g>
        <g className={cn(styles.node, styles.nodeDelayTwo)}>
          <circle cx="472" cy="355" r="7" />
          <circle cx="472" cy="355" fill="var(--ds-primary)" r="2.5" stroke="none" />
        </g>
        <g className={cn(styles.node, styles.nodeDelayThree)}>
          <circle cx="304" cy="446" r="9" />
          <circle cx="304" cy="446" fill="var(--ds-secondary)" r="3" stroke="none" />
        </g>
        <g className={styles.node}>
          <circle cx="87" cy="374" r="7" />
          <circle cx="87" cy="374" fill="var(--ds-primary)" r="2.5" stroke="none" />
        </g>
      </g>

      <circle
        className={styles.pulse}
        cx="381"
        cy="157"
        r="17"
        stroke="var(--ds-primary)"
      />
      <circle
        className={cn(styles.pulse, styles.pulseDelay)}
        cx="173"
        cy="253"
        r="15"
        stroke="var(--ds-secondary)"
      />

      <g fill="var(--ds-muted)" fontFamily="var(--font-geist-mono)" fontSize="10" letterSpacing="1.2">
        <text x="99" y="96">QUESTION</text>
        <text x="395" y="134">METHOD</text>
        <text x="486" y="392">EVIDENCE</text>
      </g>
    </svg>
  );
}
