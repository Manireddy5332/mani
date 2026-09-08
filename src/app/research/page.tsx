import {
  getResearchOverview,
  ResearchOverviewPage,
} from "@/features/research";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "Research interests and early-stage survey work by Manikanta Reddy Anugu across Generative AI adoption, LLM evaluation, retrieval, and reliability.";

export const metadata = createPublicPageMetadata({
  title: "Research",
  description,
  path: "/research",
});

export const dynamic = "force-dynamic";

export default async function ResearchPage() {
  const overview = await getResearchOverview();
  return <ResearchOverviewPage overview={overview} />;
}
