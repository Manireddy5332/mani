export const publicIndexPaths = [
  "/",
  "/about",
  "/research",
  "/projects",
  "/experience",
  "/writing",
  "/resume",
  "/contact",
] as const;

export const privateRobotPaths = [
  "/admin",
  "/api/",
  "/sign-in",
  "/access-denied",
] as const;

export function buildPublicSitemapPaths(
  projectPaths: readonly string[],
  researchPaths: readonly string[],
): string[] {
  return [
    ...new Set([
      ...publicIndexPaths,
      ...projectPaths,
      ...researchPaths,
    ]),
  ];
}
