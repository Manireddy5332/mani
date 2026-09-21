import { assertPhotoOrigin, photoIdSchema, photoMutationSchema, ProfilePhotoError } from "./policy";
import type { AdminProfilePhotoState } from "./types";

export const photoResponseHeaders = {
  "Cache-Control": "private, no-store, max-age=0",
  "CDN-Cache-Control": "no-store",
  "Vercel-CDN-Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
  "X-Robots-Tag": "noindex, nofollow, noarchive",
};

export function photoFailure(error: unknown): Response {
  const status = error instanceof ProfilePhotoError ? error.status : 503;
  const message = error instanceof ProfilePhotoError ? error.message : "Profile photo is temporarily unavailable. Please try again.";
  return Response.json({ error: message }, { status, headers: photoResponseHeaders });
}

export type PhotoHandlerDependencies = {
  authorize: (headers: Headers) => Promise<void>;
  origin: () => string;
  read: (id: string) => Promise<AdminProfilePhotoState>;
  image: (profileId: string, imageId: string) => Promise<Response>;
  upload: (profileId: string, current: string | null, request: Request) => Promise<AdminProfilePhotoState>;
  remove: (profileId: string, current: string | null) => Promise<AdminProfilePhotoState>;
};

/** Shared, directly tested boundary: authorize before parameters, body, DB or Blob. */
export async function handleAdminPhoto(request: Request, dependencies: PhotoHandlerDependencies): Promise<Response> {
  try {
    await dependencies.authorize(request.headers);
    const url = new URL(request.url);
    const id = photoIdSchema.safeParse(url.searchParams.get("profileId"));
    if (!id.success) throw new ProfilePhotoError(400, "Choose a valid profile.");
    if (request.method === "GET") {
      const imageId = url.searchParams.get("image");
      if (imageId !== null) {
        if (!photoIdSchema.safeParse(imageId).success) throw new ProfilePhotoError(404, "Photo not found.");
        return await dependencies.image(id.data, imageId);
      }
      return Response.json(await dependencies.read(id.data), { headers: photoResponseHeaders });
    }
    assertPhotoOrigin(request.headers, dependencies.origin());
    const current = request.headers.get("x-profile-photo-current");
    if (current === null) throw new ProfilePhotoError(400, "Reload this photo panel before saving.");
    const mutation = photoMutationSchema.safeParse({ profileId: id.data, expectedAvatarId: current === "none" ? null : current });
    if (!mutation.success) throw new ProfilePhotoError(400, "Reload this photo panel before saving.");
    let state: AdminProfilePhotoState;
    if (request.method === "PUT") {
      state = await dependencies.upload(id.data, mutation.data.expectedAvatarId, request);
    } else if (request.method === "DELETE") {
      if (request.headers.get("x-profile-photo-remove-confirm") !== "true") throw new ProfilePhotoError(400, "Confirm removal before deleting the profile photo.");
      state = await dependencies.remove(id.data, mutation.data.expectedAvatarId);
    } else {
      throw new ProfilePhotoError(405, "Method not permitted.");
    }
    return Response.json(state, { headers: photoResponseHeaders });
  } catch (error) {
    return photoFailure(error);
  }
}
