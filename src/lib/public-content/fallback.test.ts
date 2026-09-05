import assert from "node:assert/strict";
import test from "node:test";

import { formatPartialPeriod } from "@/lib/public-content/dates";
import { isPubliclyEligible } from "@/lib/public-content/eligibility";
import {
  LastKnownGoodStore,
  resolvePublicContent,
} from "@/lib/public-content/fallback";

test("a successful empty database result never revives fallback content", async () => {
  const store = new LastKnownGoodStore();
  const result = await resolvePublicContent({
    key: "projects:index",
    tags: ["projects"],
    load: async () => [],
    fallback: () => ["verified-static-project"],
    store,
  });

  assert.deepEqual(result, { data: [], source: "database" });
});

test("an operational failure uses last-known-good before static fallback", async () => {
  const store = new LastKnownGoodStore();
  store.remember("profile", ["profile"], { name: "Database profile" });

  const result = await resolvePublicContent({
    key: "profile",
    tags: ["profile"],
    load: async () => {
      throw new Error("unavailable");
    },
    fallback: () => ({ name: "Static profile" }),
    store,
  });

  assert.equal(result.source, "last-known-good");
  assert.equal(result.data.name, "Database profile");
});

test("tag invalidation prevents stale last-known-good content from returning", async () => {
  const store = new LastKnownGoodStore();
  store.remember("profile", ["profile"], { name: "Old database profile" });
  store.clearByTags(["profile"]);

  const result = await resolvePublicContent({
    key: "profile",
    tags: ["profile"],
    load: async () => {
      throw new Error("unavailable");
    },
    fallback: () => ({ name: "Verified static profile" }),
    store,
  });

  assert.equal(result.source, "static-fallback");
  assert.equal(result.data.name, "Verified static profile");
});

test("the process-local last-known-good store is bounded", () => {
  const store = new LastKnownGoodStore(2);
  store.remember("one", ["profile"], 1);
  store.remember("two", ["profile"], 2);
  store.remember("three", ["profile"], 3);

  assert.equal(store.read("one"), undefined);
  assert.equal(store.read("two"), 2);
  assert.equal(store.read("three"), 3);
});

test("public eligibility requires published status and a non-future timestamp", () => {
  const now = new Date("2026-08-22T12:00:00.000Z");

  assert.equal(
    isPubliclyEligible({ status: "PUBLISHED", publishedAt: null }, now),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "DRAFT", publishedAt: "2026-08-21T12:00:00.000Z" },
      now,
    ),
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
  assert.equal(
    isPubliclyEligible(
      { status: "ARCHIVED", publishedAt: "2026-08-21T12:00:00.000Z" },
      now,
    ),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: "not-a-date" },
      now,
    ),
    false,
  );
  assert.equal(
    isPubliclyEligible(
      { status: "PUBLISHED", publishedAt: now },
      now,
    ),
    true,
  );
});

test("partial periods preserve month precision without inventing a day", () => {
  assert.equal(
    formatPartialPeriod({
      startYear: 2024,
      startMonth: 6,
      endYear: null,
      endMonth: null,
      isCurrent: true,
    }),
    "Jun 2024 — Present",
  );
  assert.equal(
    formatPartialPeriod({
      startYear: 2020,
      startMonth: 6,
      endYear: 2022,
      endMonth: 7,
      isCurrent: false,
    }),
    "Jun 2020 — Jul 2022",
  );
});
