import { portfolioContent } from "@/lib/portfolio";

import type { ProfilePageData, ResumePageData } from "./types";

function getEmailAddress(mailtoUrl: string) {
  return mailtoUrl.replace(/^mailto:/, "");
}

export function getProfilePageData(): ProfilePageData {
  const { currentResearch, education, experience, identity } = portfolioContent;

  return {
    name: identity.name,
    headline: identity.headline,
    positioning: identity.positioning,
    introduction: identity.introduction,
    location: identity.location,
    progression: [
      `My professional path spans data science and AI/ML engineering. I worked as a ${experience[1].role} on a client engagement with ${experience[1].organization} and currently work as a ${experience[0].role} on a client engagement with ${experience[0].organization}.`,
      "Across these roles, my work has included machine learning, generative AI, retrieval-augmented generation, forecasting, NLP, MLOps, model monitoring, and data engineering.",
      `Alongside professional practice, I am developing an ${currentResearch.stage.toLowerCase()} ${currentResearch.format.toLowerCase()} focused on how generative AI is adopted, evaluated, and sustained in real-world organizations.`,
    ],
    experience: experience.map((entry, index) => ({
      ...entry,
      isCurrent: index === 0,
    })),
    education,
    research: {
      title: "Generative AI adoption in industry",
      href: "/research/generative-ai-adoption-in-industry",
      stage: currentResearch.stage,
      format: currentResearch.format,
      summary: currentResearch.summary,
      questions: currentResearch.questions,
      interests: portfolioContent.researchInterests,
    },
    expertise: portfolioContent.technicalExpertise,
    certifications: [],
  };
}

export function getResumePageData(): ResumePageData {
  const profile = getProfilePageData();
  const { links, selectedProject } = portfolioContent;
  const emailAddress = getEmailAddress(links.email);
  const cvRequestUrl = `${links.email}?subject=${encodeURIComponent(
    `Academic CV request for ${profile.name}`,
  )}`;

  return {
    ...profile,
    emailAddress,
    linkedInUrl: links.linkedIn,
    cvRequestUrl,
    cvDownloadUrl: null,
    project: {
      ...selectedProject,
      href: "/projects/discover-learn-and-protect",
      repositoryUrl: links.projectRepository,
    },
  };
}
