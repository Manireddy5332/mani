import assert from "node:assert/strict";
import test from "node:test";

import { performSecureSignOut } from "./sign-out";

test("secure sign-out revokes database sessions before clearing the cookie", async () => {
  const calls: string[] = [];
  const result = await performSecureSignOut({
    revokeSessions: async () => {
      calls.push("revoke");
      return {};
    },
    signOut: async () => {
      calls.push("sign-out");
      return {};
    },
  });

  assert.deepEqual(calls, ["revoke", "sign-out"]);
  assert.deepEqual(result, { ok: true });
});

test("secure sign-out fails closed when database revocation fails", async () => {
  let signOutCalled = false;
  const result = await performSecureSignOut({
    revokeSessions: async () => ({ error: new Error("unavailable") }),
    signOut: async () => {
      signOutCalled = true;
      return {};
    },
  });

  assert.equal(signOutCalled, false);
  assert.deepEqual(result, { ok: false });
});

test("secure sign-out reports cookie-clearing failures", async () => {
  const result = await performSecureSignOut({
    revokeSessions: async () => ({}),
    signOut: async () => ({ error: new Error("unavailable") }),
  });

  assert.deepEqual(result, { ok: false });
});
