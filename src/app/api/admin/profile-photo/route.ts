import { AdminAccessError, assertAdmin } from "@/lib/auth/authorization.server";
import { getServerEnvironment } from "@/lib/env.server";
import { handleAdminPhoto, type PhotoHandlerDependencies } from "@/features/profile-photo/http";
import { ProfilePhotoError } from "@/features/profile-photo/policy";
import { readAdminPhoto, readAdminPhotoImage, removeAdminPhoto, uploadAdminPhoto } from "@/features/profile-photo/service.server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const dependencies: PhotoHandlerDependencies = {
  authorize: async (headers) => {
    try { await assertAdmin(headers); } catch (error) {
      if (error instanceof AdminAccessError) {
        const status = error.code === "UNAUTHENTICATED" ? 401 : error.code === "FORBIDDEN" ? 403 : 503;
        throw new ProfilePhotoError(status, "Administrator access is required to manage the profile photo.");
      }
      throw error;
    }
  },
  origin: () => getServerEnvironment().BETTER_AUTH_URL,
  read: readAdminPhoto,
  image: readAdminPhotoImage,
  upload: uploadAdminPhoto,
  remove: removeAdminPhoto,
};

export async function GET(request: Request) { return handleAdminPhoto(request, dependencies); }
export async function PUT(request: Request) { return handleAdminPhoto(request, dependencies); }
export async function DELETE(request: Request) { return handleAdminPhoto(request, dependencies); }
