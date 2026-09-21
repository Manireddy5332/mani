import { ProfilePhotoError } from "./policy";

/** Storage and PostgreSQL cannot share a transaction. Preserve the selected photo
 * on any failure; only a definite conflict permits deleting the unused new blob. */
export async function savePhotoWithStorage<T, A>(options: {
  upload: () => Promise<A>;
  commit: (asset: A) => Promise<T>;
  discard: (asset: A) => Promise<void>;
  refresh: () => void;
  cleanupPrevious: () => Promise<void>;
}): Promise<T> {
  const asset = await options.upload();
  let result: T;
  try {
    result = await options.commit(asset);
  } catch (error) {
    if (error instanceof ProfilePhotoError && error.status === 409) {
      await options.discard(asset).catch(() => undefined);
    }
    // An uncertain commit result must never cause deletion of a possibly-live blob.
    throw error;
  }
  await refreshSavedPhoto(options);
  return result;
}

export async function removePhotoWithStorage<T>(options: {
  commit: () => Promise<T>;
  refresh: () => void;
  cleanupPrevious: () => Promise<void>;
}): Promise<T> {
  const result = await options.commit();
  await refreshSavedPhoto(options);
  return result;
}

async function refreshSavedPhoto(options: { refresh: () => void; cleanupPrevious: () => Promise<void> }) {
  try {
    options.refresh();
  } catch {
    throw new ProfilePhotoError(503, "The photo was saved, but public refresh could not complete. Reload the photo controls to confirm the saved state.");
  } finally {
    await options.cleanupPrevious().catch(() => undefined);
  }
}
