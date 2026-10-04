"use client";

import { Analytics } from "@vercel/analytics/next";

import { beforeSendPublicPageview } from "@/lib/analytics";

export function PublicAnalytics({ enabled }: { enabled: boolean }) {
  if (!enabled) return null;

  return <Analytics beforeSend={beforeSendPublicPageview} debug={false} />;
}
