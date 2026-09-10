export const siteConfig = {
  brandName: "Mani Reddy’s Portfolio",
  adminName: "Mani Reddy’s Portfolio Administrator",
  name: "Manikanta Reddy Anugu",
  shortName: "MRA",
  role: "AI/ML Engineer · Aspiring Researcher · Prospective PhD Student",
  description:
    "The academic and professional portfolio of Manikanta Reddy Anugu, an AI/ML engineer, aspiring researcher, and prospective PhD student.",
  headline:
    "An AI/ML engineer connecting professional practice with academic inquiry.",
  summary:
    "Professional experience, academic background, research interests, selected work, and the questions shaping my next stage of study.",
  // Temporary aliases keep the Phase 2 composition compatible while it is refined.
  foundationHeadline:
    "An AI/ML engineer connecting professional practice with academic inquiry.",
  foundationSummary:
    "Professional experience, academic background, research interests, selected work, and the questions shaping my next stage of study.",
} as const;

export type SiteConfig = typeof siteConfig;
