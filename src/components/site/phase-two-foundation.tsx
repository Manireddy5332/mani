import { PersonalPortfolioHome } from "@/components/site/personal-portfolio-home";
import { getStaticHomePageData } from "@/features/home/data";

/** Backward-compatible Phase 2 composition name. */
export function PhaseTwoFoundation() {
  return <PersonalPortfolioHome home={getStaticHomePageData()} />;
}
