import assert from "node:assert/strict";
import test from "node:test";

import { adminResourceKeys } from "@/features/admin/types";

import {
  allPublicContentTags,
  publicContentTags,
  publicContentTagsByAdminResource,
} from "./cache-policy";

test("every admin resource has an explicit bounded cache-invalidation policy", () => {
  assert.deepEqual(
    Object.keys(publicContentTagsByAdminResource),
    [...adminResourceKeys],
  );

  const knownTags = new Set(allPublicContentTags);
  for (const resource of adminResourceKeys) {
    const tags = publicContentTagsByAdminResource[resource];
    assert.equal(
      new Set(tags).size,
      tags.length,
      `${resource} must not invalidate the same tag twice.`,
    );
    for (const tag of tags) {
      assert.equal(
        knownTags.has(tag),
        true,
        `${resource} contains an unknown public cache tag.`,
      );
    }
  }
});

test("cross-page resources invalidate every public projection they affect", () => {
  assert.deepEqual(publicContentTagsByAdminResource.profile, allPublicContentTags);
  assert.deepEqual(publicContentTagsByAdminResource["research-projects"], [
    publicContentTags.research,
    publicContentTags.writing,
  ]);
  assert.deepEqual(publicContentTagsByAdminResource.publications, [
    publicContentTags.publications,
    publicContentTags.research,
  ]);
  assert.deepEqual(publicContentTagsByAdminResource["skill-categories"], [
    publicContentTags.skills,
  ]);
  assert.deepEqual(publicContentTagsByAdminResource.skills, [
    publicContentTags.skills,
  ]);
});

test("read-only site settings do not invalidate unrelated public caches", () => {
  assert.deepEqual(publicContentTagsByAdminResource["site-settings"], []);
});
