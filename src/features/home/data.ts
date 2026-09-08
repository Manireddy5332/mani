import { portfolioContent } from "@/lib/portfolio";

import type { HomePageData } from "./types";

export function getStaticHomePageData(): HomePageData {
  return {
    identity: portfolioContent.identity,
    links: {
      email: portfolioContent.links.email,
      linkedIn: portfolioContent.links.linkedIn,
    },
    experience: portfolioContent.experience,
    education: portfolioContent.education,
    researchInterests: portfolioContent.researchInterests,
    currentResearch: {
      href: "/research/generative-ai-adoption-in-industry",
      title: "Generative AI adoption in industry",
      ...portfolioContent.currentResearch,
    },
    selectedProject: {
      href: "/projects/discover-learn-and-protect",
      ...portfolioContent.selectedProject,
      repositoryUrl: portfolioContent.links.projectRepository,
    },
    technicalExpertise: portfolioContent.technicalExpertise,
  };
}

