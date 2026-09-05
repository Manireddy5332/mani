import assert from "node:assert/strict";
import test from "node:test";

import { isPubliclyEligible } from "@/lib/public-content/eligibility";

import {
  mapProjectRecord,
  mapProjectsPageContent,
  type PublicProjectRow,
} from "./mappers";
import { isPublicProjectSlug } from "./public-slug";

const projectRow: PublicProjectRow = {
  advisor: null,
  category: null,
  contributions: [],
  description: null,
  features: [],
  id: "project-1",
  implementation: null,
  institution: null,
  repositoryUrl: null,
  role: null,
  seoDescription: null,
  seoTitle: null,
  shortTitle: null,
  slug: "verified-project",
  summary: "A verified project summary.",
  technologies: [],
  title: "Verified project",
  type: null,
};

test("project mapping keeps nullable data optional and empty collections empty", () => {
  const project = mapProjectRecord(projectRow);
  assert.equal(project.shortTitle, "Verified project");
  assert.equal(project.type, "Project");
  assert.equal(project.repository, null);
  assert.equal(project.advisor, null);
  assert.deepEqual(project.contributions, []);
  assert.deepEqual(project.learningFeatures, []);
  assert.deepEqual(project.technologies, []);
  assert.deepEqual(mapProjectsPageContent([]), { projects: [] });
  assert.equal(
    mapProjectRecord({ ...projectRow, repositoryUrl: "javascript:alert(1)" })
      .repository,
    null,
  );
  assert.equal(
    mapProjectRecord({
      ...projectRow,
      repositoryUrl: "https://user:pass@example.test/project",
    }).repository,
    null,
  );
});

test("project slugs are bounded and fail closed", () => {
  assert.equal(isPublicProjectSlug("verified-project"), true);
  assert.equal(isPublicProjectSlug("Verified-Project"), false);
  assert.equal(isPublicProjectSlug("verified/project"), false);
  assert.equal(isPublicProjectSlug(`a${"b".repeat(120)}`), false);
});

test("project publication eligibility requires a public past timestamp", () => {
  const now = new Date("2026-08-22T12:00:00.000Z");
  assert.equal(
    isPubliclyEligible({ status: "ARCHIVED", publishedAt: now }, now),
    false,
  );
  assert.equal(
    isPubliclyEligible({ status: "PUBLISHED", publishedAt: null }, now),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: "2026-08-22T12:00:01.000Z" },
      now,
    ),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: "2026-08-22T11:59:59.000Z" },
      now,
    ),
    true,
  );
});
