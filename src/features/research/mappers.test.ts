import assert from "node:assert/strict";
import test from "node:test";

import { isPubliclyEligible } from "@/lib/public-content/eligibility";

import {
  mapPublicationSummary,
  mapResearchDetail,
  mapResearchOverview,
  type PublicResearchProjectRow,
} from "./mappers";
import { isPublicResearchSlug } from "./public-slug";

const researchRow: PublicResearchProjectRow = {
  evidenceStatus: null,
  format: null,
  id: "research-1",
  interests: [{ id: "interest-1", name: "Reliable AI" }],
  isCurrent: true,
  methodology: null,
  methodologySummary: null,
  projects: [],
  publications: [],
  questions: [{ id: "question-1", question: "What is supported?" }],
  scopeBoundary: null,
  slug: "reliable-ai-review",
  stage: "EARLY_STAGE",
  summary: "A verified early-stage review.",
  technologies: [],
  title: "Reliable AI review",
};

test("research mapping preserves honest empty and nullable states", () => {
  assert.deepEqual(mapResearchOverview(null), {
    currentDirection: null,
    interests: [],
    publications: [],
  });

  const detail = mapResearchDetail(researchRow);
  assert.equal(detail.stage, "Early-stage");
  assert.equal(detail.format, null);
  assert.equal(detail.methodology, null);
  assert.equal(detail.evidenceStatus, null);
  assert.match(detail.publicationStatus, /No formal publication record/);
  assert.match(detail.atlasSteps[4]?.description ?? "", /No public evidence/);
  assert.equal(
    mapPublicationSummary({
      arxivUrl: "javascript:alert(1)",
      authors: [],
      id: "publication-1",
      paperUrl: null,
      slug: "publication-1",
      stage: "WORKING_PAPER",
      title: "Verified title",
      venue: null,
      year: null,
    }).href,
    null,
  );
  assert.equal(
    mapPublicationSummary({
      arxivUrl: null,
      authors: [],
      id: "publication-2",
      paperUrl: "https://user:pass@example.test/paper",
      slug: "publication-2",
      stage: "WORKING_PAPER",
      title: "Verified title",
      venue: null,
      year: null,
    }).href,
    null,
  );
});

test("research slugs are bounded and fail closed", () => {
  assert.equal(isPublicResearchSlug("reliable-ai-review"), true);
  assert.equal(isPublicResearchSlug("Reliable-AI"), false);
  assert.equal(isPublicResearchSlug("reliable_ai"), false);
  assert.equal(isPublicResearchSlug(`a${"b".repeat(120)}`), false);
});

test("research publication eligibility rejects private, null, and future rows", () => {
  const now = new Date("2026-08-22T12:00:00.000Z");
  assert.equal(
    isPubliclyEligible(
      { status: "DRAFT", publishedAt: "2026-08-21T12:00:00.000Z" },
      now,
    ),
    false,
  );
  assert.equal(
    isPubliclyEligible({ status: "PUBLISHED", publishedAt: null }, now),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: "2026-08-23T12:00:00.000Z" },
      now,
    ),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: "2026-08-21T12:00:00.000Z" },
      now,
    ),
    true,
  );
});
