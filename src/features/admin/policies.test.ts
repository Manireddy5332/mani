import assert from "node:assert/strict";
import test from "node:test";

import { publicContentTags } from "@/lib/public-content/cache-policy";

import {
  decideAdminRemoval,
  getAdminInvalidationPlan,
  isPublicationStateValid,
  revisionMatches,
  resolvePublishedAt,
  withCanonicalSortOrder,
} from "./policies";

test("publication transitions create, retain, and clear timestamps safely", () => {
  const now = new Date("2026-08-22T17:30:00.000Z");
  const existing = new Date("2026-08-20T12:00:00.000Z");

  assert.equal(resolvePublishedAt("PUBLISHED", null, now), now);
  assert.equal(resolvePublishedAt("PUBLISHED", existing, now), existing);
  assert.equal(resolvePublishedAt("DRAFT", existing, now), null);
  assert.equal(resolvePublishedAt("ARCHIVED", existing, now), null);
});

test("publication-state invariant fails closed for mismatched visibility and timestamps", () => {
  const publishedAt = new Date("2026-08-22T17:30:00.000Z");
  const cases = [
    { status: "PUBLISHED", publishedAt, expected: true },
    { status: "PUBLISHED", publishedAt: null, expected: false },
    { status: "DRAFT", publishedAt: null, expected: true },
    { status: "DRAFT", publishedAt, expected: false },
    { status: "ARCHIVED", publishedAt: null, expected: true },
    { status: "ARCHIVED", publishedAt, expected: false },
  ] as const;

  for (const { status, publishedAt: timestamp, expected } of cases) {
    assert.equal(isPublicationStateValid(status, timestamp), expected);
  }
});

test("nested collections receive canonical order without mutating their input", () => {
  const original = [
    { key: "second", sortOrder: 99 },
    { key: "first", sortOrder: -1 },
  ] as const;

  const ordered = withCanonicalSortOrder(original);

  assert.deepEqual(ordered, [
    { key: "second", sortOrder: 0 },
    { key: "first", sortOrder: 1 },
  ]);
  assert.deepEqual(original, [
    { key: "second", sortOrder: 99 },
    { key: "first", sortOrder: -1 },
  ]);
  assert.notEqual(ordered[0], original[0]);
});

test("removal decisions require an exact order-independent set and explicit confirmation", () => {
  const first = "11111111-1111-4111-8111-111111111111";
  const second = "22222222-2222-4222-8222-222222222222";

  assert.equal(decideAdminRemoval([], [], false), "proceed");
  assert.equal(
    decideAdminRemoval([first, second], [second, first], false),
    "confirmation-required",
  );
  assert.equal(
    decideAdminRemoval([first, second], [second, first], true),
    "proceed",
  );
  assert.equal(decideAdminRemoval([first, second], [first], true), "mismatch");
  assert.equal(
    decideAdminRemoval([first], [first, second], true),
    "mismatch",
  );
  assert.equal(
    decideAdminRemoval([first, second], [first, first], true),
    "mismatch",
  );
});

test("revision comparison accepts only the exact current instant", () => {
  const current = new Date("2026-08-22T17:30:00.123Z");

  assert.equal(revisionMatches(current.toISOString(), current), true);
  assert.equal(revisionMatches("2026-08-22T13:30:00.123-04:00", current), true);
  assert.equal(revisionMatches("2026-08-22T17:30:00.122Z", current), false);
  assert.equal(revisionMatches("not-a-revision", current), false);
});

test("invalidation plans include only affected public and admin projections", () => {
  const id = "11111111-1111-4111-8111-111111111111";
  assert.deepEqual(getAdminInvalidationPlan("research-projects", id), {
    tags: [publicContentTags.research, publicContentTags.writing],
    publicPaths: ["/", "/research", "/about", "/resume", "/writing"],
    adminPaths: [
      "/admin/research-projects",
      `/admin/research-projects/${id}/edit`,
    ],
  });
  assert.deepEqual(getAdminInvalidationPlan("site-settings"), {
    tags: [],
    publicPaths: [],
    adminPaths: ["/admin/site-settings"],
  });
});
