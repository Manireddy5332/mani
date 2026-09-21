import { z } from "zod";

import { applicationOriginsMatch, isSecureApplicationOrigin } from "@/lib/origin-policy";
import type { PublicProfilePhoto } from "./types";

export const PHOTO_PROVIDER = "vercel-blob-private-profile-photo";
export const MAX_PHOTO_BYTES = 2 * 1024 * 1024;
export const MAX_PHOTO_PIXELS = 20_000_000;
export const PHOTO_EDGE = 960;
export const photoIdSchema = z.uuid();
export const photoMutationSchema = z.object({
  profileId: photoIdSchema,
  expectedAvatarId: photoIdSchema.nullable(),
}).strict();
export const PHOTO_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;

export class ProfilePhotoError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ProfilePhotoError";
  }
}

export type PhotoAssetView = {
  id: string;
  provider: string | null;
  mimeType: string;
  width: number | null;
  height: number | null;
};

export function toPublicProfilePhoto(asset: PhotoAssetView | null | undefined, name: string): PublicProfilePhoto | null {
  if (!asset || !photoIdSchema.safeParse(asset.id).success ||
    asset.provider !== PHOTO_PROVIDER || asset.mimeType !== "image/webp" ||
    !asset.width || !asset.height || !Number.isInteger(asset.width) || !Number.isInteger(asset.height) ||
    asset.width < 1 || asset.height < 1 || asset.width > PHOTO_EDGE || asset.height > PHOTO_EDGE) return null;
  return { src: `/profile-photo/${asset.id}`, width: asset.width, height: asset.height, alt: `Portrait of ${name}` };
}

export function assertPhotoOrigin(headers: Headers, configuredOrigin: string): void {
  const origin = headers.get("origin");
  const fetchSite = headers.get("sec-fetch-site");
  if (!origin || !isSecureApplicationOrigin(origin) ||
    !isSecureApplicationOrigin(configuredOrigin) || !applicationOriginsMatch(origin, configuredOrigin) ||
    (fetchSite !== null && fetchSite !== "same-origin")) {
    throw new ProfilePhotoError(403, "This photo request is not permitted. Refresh the admin page and try again.");
  }
}

/** Server-generated paths only: never accept a URL or a key from the browser. */
export function isOwnedPhotoKey(key: string, profileId: string): boolean {
  if (!photoIdSchema.safeParse(profileId).success) return false;
  const prefix = `profile-photos/${profileId}/`;
  return key.startsWith(prefix) && key.endsWith(".webp") &&
    photoIdSchema.safeParse(key.slice(prefix.length, -5)).success;
}
