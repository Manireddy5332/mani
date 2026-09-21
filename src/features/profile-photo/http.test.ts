import assert from "node:assert/strict";
import test from "node:test";

import { handleAdminPhoto, photoFailure, photoResponseHeaders, type PhotoHandlerDependencies } from "./http";
import { ProfilePhotoError } from "./policy";
import type { AdminProfilePhotoState } from "./types";

const profileId = "11111111-1111-4111-8111-111111111111";
const avatarId = "22222222-2222-4222-8222-222222222222";
const origin = "https://portfolio.example";
const state: AdminProfilePhotoState = { profileId, avatarId: null, photo: null, storageConfigured: true };

function dependencies(events: string[]): PhotoHandlerDependencies {
  return {
    authorize: async () => { events.push("authorize"); },
    origin: () => { events.push("origin"); return origin; },
    read: async (id) => { events.push(`read:${id}`); return state; },
    image: async (id, imageId) => { events.push(`image:${id}:${imageId}`); return new Response("mock-image", { headers: photoResponseHeaders }); },
    upload: async (id, current) => { events.push(`upload:${id}:${current}`); return state; },
    remove: async (id, current) => { events.push(`remove:${id}:${current}`); return state; },
  };
}

function request(method: string, headers: Record<string, string> = {}, query = `profileId=${profileId}`) {
  return new Request(`${origin}/api/admin/profile-photo?${query}`, { method, headers });
}

test("missing-session and non-admin requests are denied before parameters or dependencies are accessed", async () => {
  for (const status of [401, 403]) {
    for (const method of ["GET", "PUT", "DELETE"]) {
      const events: string[] = [];
      const response = await handleAdminPhoto(request(method, {}, "profileId=invalid&image=invalid"), {
        ...dependencies(events),
        authorize: async () => { events.push("denied"); throw new ProfilePhotoError(status, "Access denied."); },
      });
      assert.equal(response.status, status);
      assert.deepEqual(events, ["denied"]);
      assert.match(response.headers.get("cache-control")!, /no-store/);
    }
  }
});

test("authorized metadata and private preview reads pass only validated ids to repositories", async () => {
  const events: string[] = [];
  const response = await handleAdminPhoto(request("GET"), dependencies(events));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), state);
  assert.deepEqual(events, ["authorize", `read:${profileId}`]);
  for (const [name, value] of Object.entries(photoResponseHeaders)) assert.equal(response.headers.get(name), value);

  events.length = 0;
  const image = await handleAdminPhoto(request("GET", {}, `profileId=${profileId}&image=${avatarId}`), dependencies(events));
  assert.equal(await image.text(), "mock-image");
  assert.deepEqual(events, ["authorize", `image:${profileId}:${avatarId}`]);
});

test("invalid profile or image ids never reach storage or database reads", async () => {
  for (const [query, status] of [["profileId=invalid", 400], [`profileId=${profileId}&image=invalid`, 404], ["", 400]] as const) {
    const events: string[] = [];
    assert.equal((await handleAdminPhoto(request("GET", {}, query), dependencies(events))).status, status);
    assert.deepEqual(events, ["authorize"]);
  }
});

test("upload requires a same-origin request and explicit current-avatar revision", async () => {
  const invalidOrigins: Record<string, string>[] = [
    {},
    { origin: "https://attacker.example", "x-profile-photo-current": "none" },
    { origin, "sec-fetch-site": "cross-site", "x-profile-photo-current": "none" },
  ];
  for (const headers of invalidOrigins) {
    const events: string[] = [];
    assert.equal((await handleAdminPhoto(request("PUT", headers), dependencies(events))).status, 403);
    assert.deepEqual(events, ["authorize", "origin"]);
  }
  for (const revision of [undefined, "", "stale", "null"]) {
    const events: string[] = [];
    const headers: Record<string, string> = { origin };
    if (revision !== undefined) headers["x-profile-photo-current"] = revision;
    assert.equal((await handleAdminPhoto(request("PUT", headers), dependencies(events))).status, 400);
    assert.deepEqual(events, ["authorize", "origin"]);
  }
});

test("valid create and replace pass the expected avatar revision without profile text", async () => {
  for (const [revision, current] of [["none", null], [avatarId, avatarId]] as const) {
    const events: string[] = [];
    const response = await handleAdminPhoto(request("PUT", { origin, "x-profile-photo-current": revision }), dependencies(events));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), state);
    assert.deepEqual(events, ["authorize", "origin", `upload:${profileId}:${current}`]);
  }
});

test("remove requires explicit confirmation and preserves the revision argument", async () => {
  for (const confirmation of [undefined, "false", "1"]) {
    const events: string[] = [];
    const headers: Record<string, string> = { origin, "x-profile-photo-current": avatarId };
    if (confirmation !== undefined) headers["x-profile-photo-remove-confirm"] = confirmation;
    assert.equal((await handleAdminPhoto(request("DELETE", headers), dependencies(events))).status, 400);
    assert.deepEqual(events, ["authorize", "origin"]);
  }
  const events: string[] = [];
  assert.equal((await handleAdminPhoto(request("DELETE", {
    origin,
    "x-profile-photo-current": avatarId,
    "x-profile-photo-remove-confirm": "true",
  }), dependencies(events))).status, 200);
  assert.deepEqual(events, ["authorize", "origin", `remove:${profileId}:${avatarId}`]);
});

test("stale database revision returns a safe conflict without a successful mutation response", async () => {
  const events: string[] = [];
  const response = await handleAdminPhoto(request("PUT", { origin, "x-profile-photo-current": avatarId }), {
    ...dependencies(events),
    upload: async () => { throw new ProfilePhotoError(409, "Photo changed. Reload before saving."); },
  });
  assert.equal(response.status, 409);
  assert.deepEqual(await response.json(), { error: "Photo changed. Reload before saving." });
});

test("unsupported mutation methods do not invoke writes", async () => {
  const events: string[] = [];
  assert.equal((await handleAdminPhoto(request("POST", { origin, "x-profile-photo-current": "none" }), dependencies(events))).status, 405);
  assert.deepEqual(events, ["authorize", "origin"]);
});

test("unexpected failures redact internal details and remain uncacheable", async () => {
  const internalDetails = "MOCK_INTERNAL_STORAGE_AND_DATABASE_DETAILS";
  const response = await handleAdminPhoto(request("GET"), {
    ...dependencies([]),
    read: async () => { throw new Error(internalDetails); },
  });
  assert.equal(response.status, 503);
  const body = await response.text();
  assert.doesNotMatch(body, new RegExp(internalDetails));
  assert.match(body, /temporarily unavailable/);
  assert.match(response.headers.get("cache-control")!, /no-store/);
  assert.equal(photoFailure(undefined).status, 503);
});
