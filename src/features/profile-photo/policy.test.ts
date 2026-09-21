import assert from "node:assert/strict";
import test from "node:test";

import {
  assertPhotoOrigin,
  isOwnedPhotoKey,
  PHOTO_EDGE,
  PHOTO_PROVIDER,
  photoMutationSchema,
  ProfilePhotoError,
  toPublicProfilePhoto,
} from "./policy";

const profileId = "11111111-1111-4111-8111-111111111111";
const assetId = "22222222-2222-4222-8222-222222222222";
const asset = { id: assetId, provider: PHOTO_PROVIDER, mimeType: "image/webp", width: 480, height: 640 };

test("photo mutation accepts only a profile id and an explicit current-avatar revision", () => {
  assert.equal(photoMutationSchema.safeParse({ profileId, expectedAvatarId: null }).success, true);
  assert.equal(photoMutationSchema.safeParse({ profileId, expectedAvatarId: assetId }).success, true);
  for (const candidate of [
    { profileId },
    { profileId: "primary", expectedAvatarId: null },
    { profileId, expectedAvatarId: "old" },
    { profileId, expectedAvatarId: null, headline: "must not update text" },
    { profileId, expectedAvatarId: null, storageKey: "untrusted" },
  ]) assert.equal(photoMutationSchema.safeParse(candidate).success, false);
});

test("feature ownership requires an exact profile prefix and immutable UUID WebP key", () => {
  const key = `profile-photos/${profileId}/${assetId}.webp`;
  assert.equal(isOwnedPhotoKey(key, profileId), true);
  for (const candidate of [
    `https://example.invalid/${key}`,
    `/profile-photos/${profileId}/${assetId}.webp`,
    `profile-photos/${assetId}/${assetId}.webp`,
    `profile-photos/${profileId}/../${assetId}.webp`,
    `profile-photos/${profileId}/photo.webp`,
    `${key}?token=not-a-real-token`,
    `${key}/extra`,
    key.replace(".webp", ".png"),
  ]) assert.equal(isOwnedPhotoKey(candidate, profileId), false, candidate);
  assert.equal(isOwnedPhotoKey(key, "primary"), false);
});

test("public DTO exposes only same-origin image dimensions and accessible alt text", () => {
  const input = { ...asset, storageKey: "private-key", publicUrl: "https://private.invalid", originalFilename: "private-file", checksum: "private-checksum" };
  const dto = toPublicProfilePhoto(input, "Existing profile name");
  assert.deepEqual(dto, {
    src: `/profile-photo/${assetId}`,
    width: 480,
    height: 640,
    alt: "Portrait of Existing profile name",
  });
  assert.deepEqual(Object.keys(dto!).sort(), ["alt", "height", "src", "width"]);
});

test("unknown providers and invalid asset metadata gracefully preserve no-photo layout", () => {
  for (const input of [
    null,
    undefined,
    { ...asset, id: "not-an-id" },
    { ...asset, provider: null },
    { ...asset, provider: "external-media" },
    { ...asset, mimeType: "image/svg+xml" },
    { ...asset, width: null },
    { ...asset, height: 0 },
    { ...asset, width: -1 },
    { ...asset, height: 10.5 },
    { ...asset, width: PHOTO_EDGE + 1 },
    { ...asset, height: Number.NaN },
  ]) assert.equal(toPublicProfilePhoto(input, "Name"), null);
});

test("photo writes require the exact configured secure same origin", () => {
  const origin = "https://portfolio.example";
  assert.doesNotThrow(() => assertPhotoOrigin(new Headers({ origin, "sec-fetch-site": "same-origin" }), origin));
  assert.doesNotThrow(() => assertPhotoOrigin(new Headers({ origin }), `${origin}/`));
  assert.doesNotThrow(() => assertPhotoOrigin(new Headers({ origin: "http://localhost:3000" }), "http://localhost:3000"));
  for (const headers of [
    new Headers(),
    new Headers({ origin: "null" }),
    new Headers({ origin: "https://attacker.example", host: "portfolio.example", "x-forwarded-host": "portfolio.example" }),
    new Headers({ origin: "http://portfolio.example" }),
    new Headers({ origin: `${origin}/admin` }),
    new Headers({ origin, "sec-fetch-site": "cross-site" }),
    new Headers({ origin, "sec-fetch-site": "same-site" }),
  ]) assert.throws(() => assertPhotoOrigin(headers, origin), (error) => error instanceof ProfilePhotoError && error.status === 403);
  assert.throws(() => assertPhotoOrigin(new Headers({ origin }), "http://portfolio.example"), ProfilePhotoError);
});
