import type { Metadata } from "next";
import { notFound } from "next/navigation";

import {
  getResearchDetailBySlug,
  ResearchDetailPage,
} from "@/features/research";
import { createPublicPageMetadata } from "@/lib/metadata";

type ResearchDetailRouteProps = {
  readonly params: Promise<{ slug: string }>;
};

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: ResearchDetailRouteProps): Promise<Metadata> {
  const { slug } = await params;
  const research = await getResearchDetailBySlug(slug);

  if (!research) {
    notFound();
  }

  return createPublicPageMetadata({
    title: `${research.title} — Research direction`,
    description: research.summary,
    path: research.href,
  });
}

export default async function ResearchDetailRoute({
  params,
}: ResearchDetailRouteProps) {
  const { slug } = await params;
  const research = await getResearchDetailBySlug(slug);

  if (!research) {
    notFound();
  }

  return <ResearchDetailPage research={research} />;
}
