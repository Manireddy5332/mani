import assert from "node:assert/strict";
import test from "node:test";

import {
  decideAdminAccess,
  isAdminIdentity,
  normalizeEmail,
} from "./policy";

const configuredAdmin = "admin@example.com";

test("normalizes only whitespace and letter case", () => {
  assert.equal(normalizeEmail("  Admin@Example.COM  "), configuredAdmin);
  assert.notEqual(normalizeEmail("ad.min@example.com"), configuredAdmin);
  assert.notEqual(normalizeEmail("admin+portfolio@example.com"), configuredAdmin);
});

test("allows only the exact verified configured identity", () => {
  assert.equal(
    isAdminIdentity(
      { email: " ADMIN@example.com ", emailVerified: true },
      configuredAdmin,
    ),
    true,
  );
  assert.equal(
    isAdminIdentity(
      { email: configuredAdmin, emailVerified: false },
      configuredAdmin,
    ),
    false,
  );
  assert.equal(
    isAdminIdentity(
      { email: "someone@example.com", emailVerified: true },
      configuredAdmin,
    ),
    false,
  );
});

test("classifies missing, non-admin, and admin sessions", () => {
  assert.equal(decideAdminAccess(null, configuredAdmin), "unauthenticated");
  assert.equal(
    decideAdminAccess(
      { user: { email: "someone@example.com", emailVerified: true } },
      configuredAdmin,
    ),
    "forbidden",
  );
  assert.equal(
    decideAdminAccess(
      { user: { email: configuredAdmin, emailVerified: true } },
      configuredAdmin,
    ),
    "authorized",
  );
});

test("fails closed for incomplete identities and an empty allowlist", () => {
  for (const identity of [
    undefined,
    null,
    {},
    { email: null, emailVerified: true },
    { email: "", emailVerified: true },
    { email: configuredAdmin },
    { email: configuredAdmin, emailVerified: null },
    { email: configuredAdmin, emailVerified: false },
  ]) {
    assert.equal(isAdminIdentity(identity, configuredAdmin), false);
  }

  assert.equal(
    isAdminIdentity(
      { email: configuredAdmin, emailVerified: true },
      "   ",
    ),
    false,
  );
});

test("authorization matrix distinguishes authentication from administration", () => {
  const cases = [
    { session: null, expected: "unauthenticated" },
    { session: {}, expected: "unauthenticated" },
    { session: { user: null }, expected: "unauthenticated" },
    {
      session: { user: {} },
      expected: "forbidden",
    },
    {
      session: {
        user: { email: configuredAdmin, emailVerified: false },
      },
      expected: "forbidden",
    },
    {
      session: {
        user: { email: "admin+portfolio@example.com", emailVerified: true },
      },
      expected: "forbidden",
    },
    {
      session: {
        user: { email: " ADMIN@EXAMPLE.COM ", emailVerified: true },
      },
      expected: "authorized",
    },
  ] as const;

  for (const { session, expected } of cases) {
    assert.equal(decideAdminAccess(session, configuredAdmin), expected);
  }
});
