import assert from "node:assert/strict";
import test from "node:test";

import {
  adminNavigationGroups,
  adminResources,
  getAdminRelationResources,
  getAdminResource,
  getAdminResourcePath,
  isAdminResourceKey,
} from "./registry";
import { adminResourceKeys } from "./types";

const expectedResourceAllowlist = [
  "profile",
  "social-links",
  "research-interests",
  "research-projects",
  "projects",
  "experience",
  "education",
  "publications",
  "articles",
  "resumes",
  "skill-categories",
  "skills",
  "certifications",
  "site-settings",
] as const;

test("admin resources are limited to the explicit closed allowlist", () => {
  assert.deepEqual(adminResourceKeys, expectedResourceAllowlist);
  assert.deepEqual(
    adminResources.map((resource) => resource.key),
    expectedResourceAllowlist,
  );
  assert.equal(new Set(adminResources.map((resource) => resource.key)).size, adminResources.length);

  for (const key of expectedResourceAllowlist) {
    assert.equal(isAdminResourceKey(key), true);
    assert.equal(getAdminResource(key)?.key, key);
    assert.equal(getAdminResourcePath(key), `/admin/${key}`);
  }
});

test("resource lookup rejects unknown and path-like values", () => {
  for (const value of [
    "",
    "PROFILE",
    "unknown",
    "../profile",
    "profile/edit",
    "__proto__",
    "constructor",
    "toString",
  ]) {
    assert.equal(isAdminResourceKey(value), false);
    assert.equal(getAdminResource(value), null);
  }
});

test("registry fields never expose server-owned persistence fields", () => {
  const serverOwnedFields = new Set([
    "id",
    "profileId",
    "source",
    "seedKey",
    "createdAt",
    "updatedAt",
    "deletedAt",
  ]);

  for (const resource of adminResources) {
    for (const field of resource.fields) {
      assert.equal(
        serverOwnedFields.has(field.name),
        false,
        `${resource.key} must not expose ${field.name}.`,
      );
    }
  }
});

test("site settings remain closed until keys and schemas are explicitly allowlisted", () => {
  const siteSettings = getAdminResource("site-settings");
  assert.ok(siteSettings);
  assert.deepEqual(siteSettings.capabilities, {
    create: false,
    update: false,
    delete: false,
  });
  assert.deepEqual(siteSettings.fields, []);
});

test("authentication and private infrastructure are excluded from content management", () => {
  for (const resource of [
    "users",
    "sessions",
    "accounts",
    "verification",
    "contact-submissions",
    "media-assets",
  ]) {
    assert.equal(isAdminResourceKey(resource), false);
    assert.equal(getAdminResource(resource), null);
  }

  const privateFieldNames = new Set([
    "password",
    "token",
    "accessToken",
    "refreshToken",
    "idToken",
    "hashedIp",
    "userAgent",
  ]);
  for (const resource of adminResources) {
    for (const field of resource.fields) {
      assert.equal(
        privateFieldNames.has(field.name),
        false,
        `${resource.key} must not expose private field ${field.name}.`,
      );
    }
  }
});

test("registry relations and navigation groups are deterministic", () => {
  assert.deepEqual(adminNavigationGroups, [
    "Identity",
    "Research",
    "Portfolio",
    "Publishing",
    "Configuration",
  ]);

  const skills = getAdminResource("skills");
  assert.ok(skills);
  assert.deepEqual(getAdminRelationResources(skills), ["skill-categories"]);

  for (const resource of adminResources.filter(({ key }) => key !== "skills")) {
    assert.deepEqual(getAdminRelationResources(resource), []);
  }
});
