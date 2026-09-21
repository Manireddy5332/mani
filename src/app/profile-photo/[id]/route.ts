import { photoFailure } from "@/features/profile-photo/http";
import { photoIdSchema, ProfilePhotoError } from "@/features/profile-photo/policy";
import { readPublicPhotoImage } from "@/features/profile-photo/service.server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    if (!photoIdSchema.safeParse(id).success) throw new ProfilePhotoError(404, "Photo not found.");
    return await readPublicPhotoImage(id);
  } catch (error) {
    return photoFailure(error);
  }
}
