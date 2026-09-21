import assert from "node:assert/strict";
import test from "node:test";
import sharp from "sharp";

import { normalizeProfilePhoto, readPhotoBody } from "./image";
import { MAX_PHOTO_BYTES, PHOTO_EDGE, ProfilePhotoError } from "./policy";

function streamRequest(chunks: Uint8Array[], declaredSize?: string, onCancel?: () => void) {
  const body = new ReadableStream<Uint8Array>({
    start(controller) { for (const chunk of chunks) controller.enqueue(chunk); },
    pull(controller) { controller.close(); },
    cancel() { onCancel?.(); },
  }, { highWaterMark: 0 });
  const init: RequestInit & { duplex: "half" } = { method: "PUT", body, duplex: "half" };
  if (declaredSize !== undefined) init.headers = { "content-length": declaredSize };
  return new Request("https://portfolio.example/api/admin/profile-photo", init);
}

function photoError(status: number) {
  return (error: unknown) => error instanceof ProfilePhotoError && error.status === status;
}

test("body reader accepts bounded uploads without relying on content-length", async () => {
  const chunks = [new Uint8Array([1, 2]), new Uint8Array([3, 4])];
  assert.deepEqual(await readPhotoBody(streamRequest(chunks)), Buffer.from([1, 2, 3, 4]));
  assert.deepEqual(await readPhotoBody(streamRequest(chunks, "1")), Buffer.from([1, 2, 3, 4]));
  assert.equal((await readPhotoBody(streamRequest([new Uint8Array(MAX_PHOTO_BYTES)], String(MAX_PHOTO_BYTES)))).length, MAX_PHOTO_BYTES);
});

test("body reader rejects and cancels oversized streams with missing or lying lengths", async () => {
  for (const declaredSize of [undefined, "1"]) {
    let cancelled = false;
    await assert.rejects(readPhotoBody(streamRequest([new Uint8Array(MAX_PHOTO_BYTES), new Uint8Array(1)], declaredSize, () => { cancelled = true; })), photoError(413));
    assert.equal(cancelled, true);
  }
});

test("body reader rejects invalid declared sizes and missing request bodies", async () => {
  for (const size of [String(MAX_PHOTO_BYTES + 1), "NaN", "-1", "1.5", "1e9"]) {
    await assert.rejects(readPhotoBody(streamRequest([new Uint8Array(1)], size)), photoError(413));
  }
  await assert.rejects(readPhotoBody(new Request("https://portfolio.example", { method: "PUT" })), photoError(400));
});

test("JPEG PNG and WebP decode and normalize to bounded metadata-free WebP without enlargement", async () => {
  for (const format of ["jpeg", "png", "webp"] as const) {
    const input = await sharp({ create: { width: 40, height: 60, channels: 3, background: "#345678" } }).toFormat(format).toBuffer();
    const output = await normalizeProfilePhoto(input, `image/${format}`);
    const metadata = await sharp(output.bytes).metadata();
    assert.equal(metadata.format, "webp");
    assert.equal(metadata.width, 40);
    assert.equal(metadata.height, 60);
    assert.equal(output.width, 40);
    assert.equal(output.height, 60);
    assert.equal(metadata.exif, undefined);
    assert.equal(metadata.icc, undefined);
    assert.equal(metadata.xmp, undefined);
    assert.ok(output.bytes.length <= MAX_PHOTO_BYTES);
  }
});

test("orientation is applied before sizing and EXIF is stripped from the result", async () => {
  const input = await sharp({ create: { width: 160, height: 80, channels: 3, background: "#345678" } }).withMetadata({ orientation: 6 }).jpeg().toBuffer();
  assert.equal((await sharp(input).metadata()).orientation, 6);
  const output = await normalizeProfilePhoto(input, "image/jpeg");
  assert.equal(output.width, 80);
  assert.equal(output.height, 160);
  assert.equal((await sharp(output.bytes).metadata()).exif, undefined);
});

test("large valid portraits are resized within 960 pixels while preserving aspect ratio", async () => {
  const input = await sharp({ create: { width: 1200, height: 1800, channels: 3, background: "#345678" } }).png().toBuffer();
  const output = await normalizeProfilePhoto(input, "image/png");
  assert.equal(output.height, PHOTO_EDGE);
  assert.equal(output.width, 640);
});

test("spoofed types, malformed images, SVG, empty and oversized data fail safely", async () => {
  const png = await sharp({ create: { width: 8, height: 8, channels: 3, background: "#345678" } }).png().toBuffer();
  for (const [bytes, type] of [
    [png, "image/jpeg"],
    [png, "application/octet-stream"],
    [png, null],
    [Buffer.from("not an image"), "image/png"],
    [png.subarray(0, 32), "image/png"],
    [Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'), "image/svg+xml"],
    [Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'), "image/png"],
    [Buffer.alloc(0), "image/png"],
    [Buffer.alloc(MAX_PHOTO_BYTES + 1), "image/png"],
  ] as const) await assert.rejects(normalizeProfilePhoto(bytes, type), photoError(400));
});

test("decompression-bound images above 20 megapixels are rejected", async () => {
  const input = await sharp({ create: { width: 4500, height: 4500, channels: 3, background: "#345678" } }).png().toBuffer();
  assert.ok(input.length < MAX_PHOTO_BYTES);
  await assert.rejects(normalizeProfilePhoto(input, "image/png"), photoError(400));
});

test("decoded multi-frame WebP animations are rejected", async () => {
  const pixels = Buffer.alloc(8 * 16 * 3);
  pixels.fill(255, 0, 8 * 8 * 3);
  const input = await sharp(pixels, { raw: { width: 8, height: 16, channels: 3, pageHeight: 8 } }).webp({ loop: 0, delay: [100, 100] }).toBuffer();
  assert.equal((await sharp(input, { animated: true }).metadata()).pages, 2);
  await assert.rejects(normalizeProfilePhoto(input, "image/webp"), photoError(400));
});

function crc32(bytes: Buffer): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

test("APNG animation control is rejected even if a decoder exposes only its static fallback", async () => {
  const png = await sharp({ create: { width: 8, height: 8, channels: 3, background: "#345678" } }).png().toBuffer();
  const animation = Buffer.alloc(20);
  animation.writeUInt32BE(8, 0);
  animation.write("acTL", 4, "ascii");
  animation.writeUInt32BE(2, 8);
  animation.writeUInt32BE(0, 12);
  animation.writeUInt32BE(crc32(animation.subarray(4, 16)), 16);
  const input = Buffer.concat([png.subarray(0, 33), animation, png.subarray(33)]);
  await assert.rejects(normalizeProfilePhoto(input, "image/png"), photoError(400));
});
