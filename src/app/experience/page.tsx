import {
  ExperiencePage,
  getExperiencePageContent,
} from "@/features/experience";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "Professional experience, education, and technical expertise from the academic portfolio of Manikanta Reddy Anugu.";

export const metadata = createPublicPageMetadata({
  title: "Experience",
  description,
  path: "/experience",
});

export const dynamic = "force-dynamic";

export default async function ExperienceRoute() {
  return <ExperiencePage content={await getExperiencePageContent()} />;
}
