import assert from "node:assert/strict";
import test from "node:test";

import { removePhotoWithStorage, savePhotoWithStorage } from "./flow";
import { ProfilePhotoError } from "./policy";

function saveOptions(events: string[]) {
  return {
    upload: async () => { events.push("upload-new"); return "new-private-asset"; },
    commit: async (asset: string) => { events.push(`commit:${asset}`); return "saved"; },
    discard: async (asset: string) => { events.push(`discard:${asset}`); },
    refresh: () => { events.push("refresh"); },
    cleanupPrevious: async () => { events.push("cleanup-previous"); },
  };
}

test("replacement uploads before committing then invalidates before previous-photo cleanup", async () => {
  const events: string[] = [];
  assert.equal(await savePhotoWithStorage(saveOptions(events)), "saved");
  assert.deepEqual(events, ["upload-new", "commit:new-private-asset", "refresh", "cleanup-previous"]);
});

test("upload failure cannot alter the selected photo, invalidate caches, or delete assets", async () => {
  const events: string[] = [];
  const failure = new Error("mock upload failure");
  await assert.rejects(savePhotoWithStorage({ ...saveOptions(events), upload: async () => { events.push("upload-new"); throw failure; } }), (error) => error === failure);
  assert.deepEqual(events, ["upload-new"]);
});

test("definite revision conflict discards only the unused new blob", async () => {
  const events: string[] = [];
  const failure = new ProfilePhotoError(409, "Photo changed. Reload before saving.");
  await assert.rejects(savePhotoWithStorage({ ...saveOptions(events), commit: async () => { events.push("conflict"); throw failure; } }), (error) => error === failure);
  assert.deepEqual(events, ["upload-new", "conflict", "discard:new-private-asset"]);
});

test("cleanup failure after a conflict does not mask the revision conflict", async () => {
  const events: string[] = [];
  const failure = new ProfilePhotoError(409, "Photo changed.");
  await assert.rejects(savePhotoWithStorage({ ...saveOptions(events), commit: async () => { throw failure; }, discard: async () => { throw new Error("mock cleanup failure"); } }), (error) => error === failure);
  assert.deepEqual(events, ["upload-new"]);
});

test("uncertain database commit never deletes a possibly committed new photo", async () => {
  for (const failure of [new Error("mock response lost after commit"), new ProfilePhotoError(503, "Unavailable")]) {
    const events: string[] = [];
    await assert.rejects(savePhotoWithStorage({ ...saveOptions(events), commit: async () => { events.push("uncertain-commit"); throw failure; } }), (error) => error === failure);
    assert.deepEqual(events, ["upload-new", "uncertain-commit"]);
  }
});

test("old-blob cleanup failure cannot fail an already saved replacement", async () => {
  const events: string[] = [];
  assert.equal(await savePhotoWithStorage({ ...saveOptions(events), cleanupPrevious: async () => { events.push("cleanup-failed"); throw new Error("mock cleanup failure"); } }), "saved");
  assert.deepEqual(events, ["upload-new", "commit:new-private-asset", "refresh", "cleanup-failed"]);
});

test("remove detaches the selected avatar then invalidates before storage cleanup", async () => {
  const events: string[] = [];
  const result = await removePhotoWithStorage({
    commit: async () => { events.push("detach"); return { avatarId: null }; },
    refresh: () => { events.push("refresh"); },
    cleanupPrevious: async () => { events.push("cleanup-previous"); },
  });
  assert.deepEqual(result, { avatarId: null });
  assert.deepEqual(events, ["detach", "refresh", "cleanup-previous"]);
});

test("failed remove cannot invalidate or clean up the still-selected photo", async () => {
  const events: string[] = [];
  await assert.rejects(removePhotoWithStorage({
    commit: async () => { events.push("detach-failed"); throw new ProfilePhotoError(409, "Photo changed."); },
    refresh: () => { events.push("refresh"); },
    cleanupPrevious: async () => { events.push("cleanup-previous"); },
  }), ProfilePhotoError);
  assert.deepEqual(events, ["detach-failed"]);
});

test("storage cleanup failure cannot fail a successful photo removal", async () => {
  assert.equal(await removePhotoWithStorage({
    commit: async () => "removed",
    refresh: () => undefined,
    cleanupPrevious: async () => { throw new Error("mock cleanup failure"); },
  }), "removed");
});

test("failed refresh after replacement reports saved state and never discards the new photo", async () => {
  const events: string[] = [];
  await assert.rejects(savePhotoWithStorage({
    ...saveOptions(events),
    refresh: () => { events.push("refresh-failed"); throw new Error("mock cache failure"); },
  }), (error) => error instanceof ProfilePhotoError && error.status === 503 && /saved.*refresh/i.test(error.message));
  assert.deepEqual(events, ["upload-new", "commit:new-private-asset", "refresh-failed", "cleanup-previous"]);
});

test("failed refresh after removal still cleans up detached storage and reports saved state", async () => {
  const events: string[] = [];
  await assert.rejects(removePhotoWithStorage({
    commit: async () => { events.push("detach"); return "removed"; },
    refresh: () => { events.push("refresh-failed"); throw new Error("mock cache failure"); },
    cleanupPrevious: async () => { events.push("cleanup-previous"); },
  }), (error) => error instanceof ProfilePhotoError && error.status === 503 && /saved.*refresh/i.test(error.message));
  assert.deepEqual(events, ["detach", "refresh-failed", "cleanup-previous"]);
});
