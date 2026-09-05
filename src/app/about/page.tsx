import { notFound } from "next/navigation";

import { AboutPage } from "@/features/profile";
import { getPublicProfilePageData } from "@/features/profile/queries.server";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "Learn about Manikanta Reddy Anugu's AI/ML engineering background, academic foundation, and developing research direction.";

export const metadata = createPublicPageMetadata({
  title: "About",
  description,
  path: "/about",
});

export const dynamic = "force-dynamic";

export default async function AboutRoute() {
  const profile = await getPublicProfilePageData();
  if (!profile) notFound();
  return <AboutPage profile={profile} />;
}
