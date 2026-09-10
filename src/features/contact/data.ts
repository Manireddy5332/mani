import { portfolioContent } from "@/lib/portfolio";

import type { ContactPageData } from "./types";

export const contactIntroduction =
  "For academic conversations, research opportunities, or professional AI/ML discussions, email and LinkedIn are the best ways to reach me.";

export function getContactPageData(): ContactPageData {
  const { identity, links } = portfolioContent;
  const emailAddress = links.email.replace(/^mailto:/, "");

  return {
    name: identity.name,
    introduction: contactIntroduction,
    methods: [
      {
        key: "email",
        kind: "email",
        label: "Email",
        value: emailAddress,
        href: links.email,
        description: "Direct correspondence for academic and professional inquiries.",
      },
      {
        key: "linkedin",
        kind: "linkedin",
        label: "LinkedIn",
        value: "Professional profile",
        href: links.linkedIn,
        external: true,
        description: "Professional background and networking.",
      },
      {
        key: "location",
        kind: "location",
        label: "Location",
        value: identity.location,
        description: "Current base for professional and academic work.",
      },
      {
        key: "project-repository",
        kind: "repository",
        label: "Project repository",
        value: "IT Capstone on GitHub",
        href: links.projectRepository,
        external: true,
        description: "Repository for the academic capstone project.",
      },
    ],
  };
}

export const getStaticContactPageData = getContactPageData;
