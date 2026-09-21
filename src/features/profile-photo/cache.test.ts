import assert from "node:assert/strict";
import test from "node:test";
import { hasLocalMatch } from "next/dist/shared/lib/match-local-pattern";

import nextConfig from "../../../next.config";

test("shared image optimizer cannot retain publication-gated profile photos or admin previews", () => {
  const patterns = nextConfig.images?.localPatterns;
  assert.ok(patterns, "The optimizer must have an explicit local-image allowlist.");
  for (const url of [
    "/profile-photo/22222222-2222-4222-8222-222222222222",
    "/profile-photo/22222222-2222-4222-8222-222222222222?cache=1",
    "/api/admin/profile-photo?profileId=11111111-1111-4111-8111-111111111111&image=22222222-2222-4222-8222-222222222222",
    "/admin/profile-photo",
  ]) assert.equal(hasLocalMatch(patterns, url), false, url);
});

test("static build-time media remains eligible for image optimization", () => {
  assert.equal(hasLocalMatch(nextConfig.images?.localPatterns, "/_next/static/media/portrait.12345678.png"), true);
});
