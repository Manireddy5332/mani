import type { Metadata } from "next";
import { notFound } from "next/navigation";

import {
  getProjectDetailBySlug,
  ProjectDetailPage,
} from "@/features/projects";
import { createPublicPageMetadata } from "@/lib/metadata";

type ProjectDetailRouteProps = {
  readonly params: Promise<{ slug: string }>;
};

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: ProjectDetailRouteProps): Promise<Metadata> {
  const { slug } = await params;
  const project = await getProjectDetailBySlug(slug);

  if (!project) {
    notFound();
  }

  return createPublicPageMetadata({
    title: project.seoTitle ?? `${project.shortTitle} — Project case study`,
    description: project.seoDescription ?? project.summary,
    path: project.href,
  });
}

export default async function ProjectDetailRoute({
  params,
}: ProjectDetailRouteProps) {
  const { slug } = await params;
  const project = await getProjectDetailBySlug(slug);

  if (!project) {
    notFound();
  }

  return <ProjectDetailPage project={project} />;
}
