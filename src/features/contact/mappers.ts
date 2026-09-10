import { contactIntroduction } from "./data";
import type { ContactMethod, ContactPageData } from "./types";

export type PublicContactRecord = Readonly<{
  name: string;
  location: string | null;
  socialLinks: readonly Readonly<{
    key: string;
    kind: string;
    label: string;
    url: string;
    handle: string | null;
  }>[];
  projects: readonly Readonly<{
    title: string;
    shortTitle: string | null;
    repositoryUrl: string | null;
  }>[];
}>;

function safeUrl(value: string, kind: string): string | null {
  try {
    const url = new URL(value);
    if (kind === "EMAIL") {
      return url.protocol === "mailto:" && url.pathname.trim() ? value : null;
    }
    return (url.protocol === "https:" || url.protocol === "http:") &&
      !url.username &&
      !url.password
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

function socialKind(kind: string): ContactMethod["kind"] {
  if (kind === "EMAIL") return "email";
  if (kind === "LINKEDIN") return "linkedin";
  if (kind === "GITHUB") return "github";
  if (kind === "WEBSITE") return "website";
  return "other";
}

function socialDescription(kind: ContactMethod["kind"]): string {
  if (kind === "email") {
    return "Direct correspondence for academic and professional inquiries.";
  }
  if (kind === "linkedin") return "Professional background and networking.";
  if (kind === "github") return "Public code and project repositories.";
  if (kind === "website") return "An additional public professional website.";
  return "An additional verified professional link.";
}

function mapSocialLink(
  link: PublicContactRecord["socialLinks"][number],
): ContactMethod | null {
  const href = safeUrl(link.url, link.kind);
  if (!href) return null;
  const kind = socialKind(link.kind);
  let value = link.handle ?? link.label;
  if (kind === "email") {
    try {
      value = decodeURIComponent(new URL(href).pathname);
    } catch {
      value = link.label;
    }
  }
  if (kind === "linkedin" && !link.handle) value = "Professional profile";

  return {
    key: link.key,
    kind,
    label: link.label,
    value,
    href,
    external: kind !== "email",
    description: socialDescription(kind),
  };
}

export function mapPublicContactRecord(
  record: PublicContactRecord,
): ContactPageData {
  const methods = record.socialLinks
    .map(mapSocialLink)
    .filter((method): method is ContactMethod => method !== null);

  if (record.location) {
    methods.push({
      key: "location",
      kind: "location",
      label: "Location",
      value: record.location,
      description: "Current base for professional and academic work.",
    });
  }

  const project = record.projects[0];
  const repositoryUrl = project?.repositoryUrl
    ? safeUrl(project.repositoryUrl, "PROJECT")
    : null;
  if (project && repositoryUrl) {
    methods.push({
      key: "project-repository",
      kind: "repository",
      label: "Project repository",
      value: project.shortTitle ?? project.title,
      href: repositoryUrl,
      external: true,
      description: "Repository for the selected academic project.",
    });
  }

  return { name: record.name, introduction: contactIntroduction, methods };
}
