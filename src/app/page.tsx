import { notFound } from "next/navigation";

import { PersonalPortfolioHome } from "@/components/site/personal-portfolio-home";
import { getPublicHomePageData } from "@/features/home/queries.server";

export const dynamic = "force-dynamic";

export default async function Home() {
  const home = await getPublicHomePageData();
  if (!home) notFound();
  return <PersonalPortfolioHome home={home} />;
}
