import assert from "node:assert/strict";
import test from "node:test";

import {
  createPublicPageMetadata,
  normalizeMetadataText,
} from "./metadata";

test("normalizes whitespace and bounds metadata without changing short copy", () => {
  assert.equal(normalizeMetadataText("  A short\n title  ", 70), "A short title");
  assert.equal(normalizeMetadataText("abcdefgh", 5), "abcd…");
});

test("public metadata keeps canonical paths and bounded descriptions", () => {
  const metadata = createPublicPageMetadata({
    title: "A".repeat(100),
    description: "B".repeat(200),
    path: "/research/verified-direction",
  });

  assert.equal(String(metadata.title).length, 70);
  assert.equal(String(metadata.description).length, 160);
  assert.deepEqual(metadata.alternates, {
    canonical: "/research/verified-direction",
  });
});
