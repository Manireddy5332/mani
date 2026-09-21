import "server-only";

import { getDatabase } from "@/lib/db";
import { PHOTO_PROVIDER, ProfilePhotoError, isOwnedPhotoKey } from "./policy";

const avatarSelect = { id: true, provider: true, mimeType: true, width: true, height: true, storageKey: true } as const;
const profileSelect = { id: true, name: true, avatarId: true, avatar: { select: avatarSelect } } as const;

export async function readPhotoProfile(profileId: string) {
  const profile = await getDatabase().profile.findFirst({ where: { id: profileId, key: "primary" }, select: profileSelect });
  if (!profile) throw new ProfilePhotoError(404, "Photo management is available for the primary profile only.");
  return profile;
}

/** Deliberately uncached: a stale public page must not serve a removed/private photo. */
export async function readPublishedPhoto(assetId: string) {
  return getDatabase().profile.findFirst({
    where: { key: "primary", status: "PUBLISHED", avatarId: assetId },
    select: profileSelect,
  });
}

export type NewPhotoAsset = { id: string; storageKey: string; width: number; height: number; byteSize: number };

export async function changeProfilePhoto(profileId: string, expectedAvatarId: string | null, asset: NewPhotoAsset | null) {
  return getDatabase().$transaction(async (transaction) => {
    if (asset) {
      await transaction.mediaAsset.create({ data: {
        ...asset,
        byteSize: BigInt(asset.byteSize),
        provider: PHOTO_PROVIDER,
        originalFilename: "profile-photo.webp",
        mimeType: "image/webp",
        // Private storage URL and original EXIF/filename are intentionally not persisted.
      } });
    }
    const result = await transaction.profile.updateMany({
      where: { id: profileId, key: "primary", avatarId: expectedAvatarId },
      data: { avatarId: asset?.id ?? null },
    });
    if (result.count !== 1) throw new ProfilePhotoError(409, "The profile photo changed in another tab. Reload this photo panel before saving again.");
    return transaction.profile.findUniqueOrThrow({ where: { id: profileId }, select: profileSelect });
  });
}

/** Claim only a detached, feature-owned asset; never delete shared CMS media. */
export async function detachUnusedPhotoMetadata(profileId: string, assetId: string): Promise<string | null> {
  const database = getDatabase();
  const asset = await database.mediaAsset.findUnique({ where: { id: assetId }, select: { storageKey: true, provider: true } });
  if (!asset || asset.provider !== PHOTO_PROVIDER || !isOwnedPhotoKey(asset.storageKey, profileId)) return null;
  const result = await database.mediaAsset.deleteMany({ where: {
    id: assetId, provider: PHOTO_PROVIDER, storageKey: asset.storageKey,
    avatarFor: { is: null }, certificationFor: { is: null }, resumeFor: { is: null },
    projectPlacements: { none: {} }, researchPlacements: { none: {} }, articlePlacements: { none: {} },
  } });
  return result.count === 1 ? asset.storageKey : null;
}
