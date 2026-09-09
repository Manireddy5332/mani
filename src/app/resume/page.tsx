import { notFound } from "next/navigation";

import { ResumePage } from "@/features/profile";
import { getPublicResumePageData } from "@/features/profile/queries.server";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "A structured overview of Manikanta Reddy Anugu's professional experience, education, research direction, academic project, and technical expertise.";

export const metadata = createPublicPageMetadata({
  title: "Academic CV",
  description,
  path: "/resume",
});

export const dynamic = "force-dynamic";

export default async function ResumeRoute() {
  const resume = await getPublicResumePageData();
  if (!resume) notFound();
  return <ResumePage resume={resume} />;
}
