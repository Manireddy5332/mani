import "server-only";

import { del, get, put } from "@vercel/blob";

import { ProfilePhotoError, isOwnedPhotoKey } from "./policy";

export function isPhotoStorageConfigured(): boolean {
  return Boolean((process.env.BLOB_STORE_ID?.trim() && process.env.VERCEL_OIDC_TOKEN?.trim()) || process.env.BLOB_READ_WRITE_TOKEN?.trim());
}

export function assertPhotoStorage(): void {
  if (!isPhotoStorageConfigured()) throw new ProfilePhotoError(503, "Private profile photo storage has not been configured yet.");
}

export async function putProfilePhoto(profileId: string, key: string, bytes: Buffer): Promise<void> {
  assertPhotoStorage();
  if (!isOwnedPhotoKey(key, profileId)) throw new ProfilePhotoError(400, "Invalid photo destination.");
  // SDK-managed OIDC on Vercel; its standard read-write token fallback locally.
  // No credentials, storage URLs, or original filenames are returned to the UI.
  await put(key, bytes, {
    access: "private", contentType: "image/webp", addRandomSuffix: false, allowOverwrite: false,
  });
}

export async function deleteProfilePhoto(profileId: string, key: string): Promise<void> {
  assertPhotoStorage();
  if (!isOwnedPhotoKey(key, profileId)) return;
  await del(key);
}

export async function getProfilePhoto(profileId: string, key: string) {
  assertPhotoStorage();
  if (!isOwnedPhotoKey(key, profileId)) throw new ProfilePhotoError(404, "Photo not found.");
  const result = await get(key, { access: "private", useCache: false });
  if (!result || result.statusCode !== 200 || result.blob.contentType !== "image/webp") {
    throw new ProfilePhotoError(404, "Photo not found.");
  }
  return result.stream;
}
