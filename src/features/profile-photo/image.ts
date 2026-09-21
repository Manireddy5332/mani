// Node-only image decoder. Imported by the server storage boundary, never a UI.
import sharp from "sharp";

import { MAX_PHOTO_BYTES, MAX_PHOTO_PIXELS, PHOTO_CONTENT_TYPES, PHOTO_EDGE, ProfilePhotoError } from "./policy";

function hasPngAnimation(bytes: Buffer): boolean {
  if (!bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return false;
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const length = bytes.readUInt32BE(offset);
    if (length > bytes.length - offset - 12) throw new Error("Invalid PNG chunk");
    const type = bytes.toString("ascii", offset + 4, offset + 8);
    if (type === "acTL") return true;
    if (type === "IEND") break;
    offset += length + 12;
  }
  return false;
}

export async function readPhotoBody(request: Request): Promise<Buffer> {
  const declaredSize = request.headers.get("content-length");
  if (declaredSize && (!/^\d+$/.test(declaredSize) || Number(declaredSize) > MAX_PHOTO_BYTES)) {
    throw new ProfilePhotoError(413, "Choose an image no larger than 2 MiB.");
  }
  if (!request.body) throw new ProfilePhotoError(400, "Choose a JPEG, PNG, or WebP image.");
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_PHOTO_BYTES) {
        await reader.cancel();
        throw new ProfilePhotoError(413, "Choose an image no larger than 2 MiB.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks, size);
}

export async function normalizeProfilePhoto(bytes: Buffer, contentType: string | null) {
  if (!bytes.length || bytes.length > MAX_PHOTO_BYTES ||
    !PHOTO_CONTENT_TYPES.includes(contentType as typeof PHOTO_CONTENT_TYPES[number])) {
    throw new ProfilePhotoError(400, "Choose a JPEG, PNG, or WebP image no larger than 2 MiB.");
  }
  try {
    // Some PNG decoders expose only APNG's first frame; reject its animation
    // control chunk explicitly rather than silently accepting an animation.
    if (hasPngAnimation(bytes)) throw new Error("Animated PNG");
    const decoder = sharp(bytes, { limitInputPixels: MAX_PHOTO_PIXELS, failOn: "warning", animated: true });
    const metadata = await decoder.metadata();
    const expectedType = { jpeg: "image/jpeg", png: "image/png", webp: "image/webp" }[metadata.format as "jpeg" | "png" | "webp"];
    if (expectedType !== contentType || !metadata.width || !metadata.height ||
      metadata.width * metadata.height > MAX_PHOTO_PIXELS || (metadata.pages ?? 1) !== 1) throw new Error("Invalid image");
    // rotate() applies EXIF orientation; default output omits EXIF/GPS metadata.
    const { data, info } = await decoder.rotate().resize({ width: PHOTO_EDGE, height: PHOTO_EDGE, fit: "inside", withoutEnlargement: true })
      .webp({ quality: 85 }).toBuffer({ resolveWithObject: true });
    if (data.length > MAX_PHOTO_BYTES) throw new Error("Image too large");
    return { bytes: data, width: info.width, height: info.height };
  } catch {
    throw new ProfilePhotoError(400, "This image could not be read. Use a non-animated JPEG, PNG, or WebP up to 20 megapixels.");
  }
}
