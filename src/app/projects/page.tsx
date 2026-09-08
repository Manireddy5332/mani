import {
  getProjectsPageContent,
  ProjectsPage,
} from "@/features/projects";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "Selected academic project work by Manikanta Reddy Anugu, including a Kennesaw State University IT capstone in biodiversity and environmental learning.";

export const metadata = createPublicPageMetadata({
  title: "Projects",
  description,
  path: "/projects",
});

export const dynamic = "force-dynamic";

export default async function ProjectsRoute() {
  const content = await getProjectsPageContent();
  return <ProjectsPage content={content} />;
}
