import "server-only";

import { randomUUID } from "node:crypto";
import { revalidatePath, revalidateTag } from "next/cache";

import { getAdminInvalidationPlan } from "@/features/admin/policies";
import { clearPublicLastKnownGood } from "@/lib/public-content/fallback.server";
import { removePhotoWithStorage, savePhotoWithStorage } from "./flow";
import { photoResponseHeaders } from "./http";
import { normalizeProfilePhoto, readPhotoBody } from "./image";
import { ProfilePhotoError, toPublicProfilePhoto } from "./policy";
import { changeProfilePhoto, detachUnusedPhotoMetadata, readPhotoProfile, readPublishedPhoto } from "./repository.server";
import { assertPhotoStorage, deleteProfilePhoto, getProfilePhoto, isPhotoStorageConfigured, putProfilePhoto } from "./storage.server";
import type { AdminProfilePhotoState } from "./types";

type PhotoProfile = Awaited<ReturnType<typeof readPhotoProfile>>;

function adminState(profile: PhotoProfile): AdminProfilePhotoState {
  const photo = toPublicProfilePhoto(profile.avatar, profile.name);
  return {
    profileId: profile.id, avatarId: profile.avatarId, storageConfigured: isPhotoStorageConfigured(),
    photo: photo ? { ...photo, src: `/api/admin/profile-photo?profileId=${profile.id}&image=${profile.avatarId}` } : null,
  };
}

function refreshPhoto(profileId: string): void {
  const plan = getAdminInvalidationPlan("profile", profileId);
  clearPublicLastKnownGood(plan.tags);
  for (const tag of plan.tags) revalidateTag(tag, { expire: 0 });
  revalidatePath("/admin", "layout");
  for (const path of [...plan.adminPaths, ...plan.publicPaths]) revalidatePath(path);
}

async function cleanupPrevious(profileId: string, assetId: string | null) {
  if (!assetId) return;
  const key = await detachUnusedPhotoMetadata(profileId, assetId);
  if (key) await deleteProfilePhoto(profileId, key);
}

function assertCurrentPhoto(profile: PhotoProfile, expected: string | null) {
  if (profile.avatarId !== expected) throw new ProfilePhotoError(409, "The profile photo changed in another tab. Reload this photo panel before saving again.");
}

export async function readAdminPhoto(profileId: string) {
  return adminState(await readPhotoProfile(profileId));
}

export async function uploadAdminPhoto(profileId: string, expected: string | null, request: Request) {
  assertPhotoStorage();
  const current = await readPhotoProfile(profileId);
  assertCurrentPhoto(current, expected);
  const image = await normalizeProfilePhoto(await readPhotoBody(request), request.headers.get("content-type"));
  const id = randomUUID();
  const asset = { id, storageKey: `profile-photos/${profileId}/${id}.webp`, width: image.width, height: image.height, byteSize: image.bytes.length };
  const result = await savePhotoWithStorage({
    upload: async () => { await putProfilePhoto(profileId, asset.storageKey, image.bytes); return asset; },
    commit: (uploaded) => changeProfilePhoto(profileId, expected, uploaded),
    discard: (uploaded) => deleteProfilePhoto(profileId, uploaded.storageKey),
    refresh: () => refreshPhoto(profileId),
    cleanupPrevious: () => cleanupPrevious(profileId, expected),
  });
  return adminState(result);
}

export async function removeAdminPhoto(profileId: string, expected: string | null) {
  assertPhotoStorage();
  const current = await readPhotoProfile(profileId);
  assertCurrentPhoto(current, expected);
  if (!expected) return adminState(current);
  const result = await removePhotoWithStorage({
    commit: () => changeProfilePhoto(profileId, expected, null),
    refresh: () => refreshPhoto(profileId),
    cleanupPrevious: () => cleanupPrevious(profileId, expected),
  });
  return adminState(result);
}

async function imageResponse(profile: PhotoProfile | null, assetId: string) {
  if (!profile || profile.avatarId !== assetId || !profile.avatar || !toPublicProfilePhoto(profile.avatar, profile.name)) {
    throw new ProfilePhotoError(404, "Photo not found.");
  }
  const stream = await getProfilePhoto(profile.id, profile.avatar.storageKey);
  return new Response(stream, { headers: {
    ...photoResponseHeaders, "Content-Type": "image/webp", "Content-Disposition": 'inline; filename="profile-photo.webp"',
  } });
}

export async function readAdminPhotoImage(profileId: string, assetId: string) {
  return imageResponse(await readPhotoProfile(profileId), assetId);
}

export async function readPublicPhotoImage(assetId: string) {
  return imageResponse(await readPublishedPhoto(assetId), assetId);
}
