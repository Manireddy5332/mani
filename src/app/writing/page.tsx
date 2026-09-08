import { getWritingOverview, WritingOverviewPage } from "@/features/writing";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "The future writing and research-notes archive of Manikanta Reddy Anugu, currently grounded in an early-stage Generative AI literature and case-study review.";

export const metadata = createPublicPageMetadata({
  title: "Writing",
  description,
  path: "/writing",
});

export const dynamic = "force-dynamic";

export default async function WritingPage() {
  return <WritingOverviewPage overview={await getWritingOverview()} />;
}
