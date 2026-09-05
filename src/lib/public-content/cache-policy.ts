export const PUBLIC_CONTENT_CACHE_SECONDS = 300;

export const publicContentTags = {
  profile: "public-content:profile",
  socialLinks: "public-content:social-links",
  research: "public-content:research",
  projects: "public-content:projects",
  experience: "public-content:experience",
  education: "public-content:education",
  publications: "public-content:publications",
  writing: "public-content:writing",
  resumes: "public-content:resumes",
  skills: "public-content:skills",
  certifications: "public-content:certifications",
} as const;

export type PublicContentTag =
  (typeof publicContentTags)[keyof typeof publicContentTags];

export const allPublicContentTags = Object.values(publicContentTags);

export const publicContentTagsByAdminResource = {
  profile: allPublicContentTags,
  "social-links": [publicContentTags.socialLinks],
  "research-interests": [publicContentTags.research],
  "research-projects": [publicContentTags.research, publicContentTags.writing],
  projects: [publicContentTags.projects],
  experience: [publicContentTags.experience],
  education: [publicContentTags.education],
  publications: [publicContentTags.publications, publicContentTags.research],
  articles: [publicContentTags.writing],
  resumes: [publicContentTags.resumes],
  "skill-categories": [publicContentTags.skills],
  skills: [publicContentTags.skills],
  certifications: [publicContentTags.certifications],
  "site-settings": [],
} as const;
