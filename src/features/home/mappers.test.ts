import assert from "node:assert/strict";
import test from "node:test";

import { getStaticHomePageData } from "./data";
import { mapPublicHomeRecord, type PublicHomeRecord } from "./mappers";

const baseRecord: PublicHomeRecord = {
  name: "Verified person",
  positioning: ["AI/ML Engineer"],
  introduction: "Verified introduction.",
  location: null,
  avatar: null,
  socialLinks: [],
  experiences: [],
  educationRecords: [],
  researchInterests: [],
  researchProjects: [],
  projects: [],
  skillCategories: [],
};

test("home mapper preserves empty published collections without static claims", () => {
  const mapped = mapPublicHomeRecord(baseRecord);

  assert.equal(mapped.currentResearch, null);
  assert.equal(mapped.selectedProject, null);
  assert.deepEqual(mapped.experience, []);
  assert.deepEqual(mapped.education, []);
  assert.deepEqual(mapped.technicalExpertise, []);
  assert.equal(mapped.identity.photo, null);
});

test("home photo DTO uses only the controlled delivery URL and display fields", () => {
  const asset = {
    id: "693a19d8-d5db-4a57-b99b-da37849a5a25",
    provider: "vercel-blob-private-profile-photo",
    mimeType: "image/webp",
    width: 640,
    height: 800,
    storageKey: "private-storage-key-not-for-public-output",
    publicUrl: "https://untrusted.example/not-used.webp",
    originalFilename: "not-for-public-output.webp",
  };
  const mapped = mapPublicHomeRecord({ ...baseRecord, avatar: asset });

  assert.equal(mapped.identity.photo?.src, `/profile-photo/${asset.id}`);
  assert.equal(mapped.identity.photo?.width, 640);
  assert.equal(mapped.identity.photo?.height, 800);
  assert.deepEqual(Object.keys(mapped.identity.photo ?? {}).sort(), [
    "alt",
    "height",
    "src",
    "width",
  ]);
  assert.doesNotMatch(
    JSON.stringify(mapped),
    /private-storage-key|untrusted\.example|not-for-public-output/,
  );
});

test("home omits unsupported photo assets and static fallback has no photo", () => {
  const mapped = mapPublicHomeRecord({
    ...baseRecord,
    avatar: {
      id: "693a19d8-d5db-4a57-b99b-da37849a5a25",
      provider: "unmanaged",
      mimeType: "image/svg+xml",
      width: 200,
      height: 200,
    },
  });

  assert.equal(mapped.identity.photo, null);
  assert.equal(getStaticHomePageData().identity.photo, null);
});

test("home mapper preserves partial dates and omits unsafe public URLs", () => {
  const mapped = mapPublicHomeRecord({
    ...baseRecord,
    socialLinks: [
      { kind: "EMAIL", url: "mailto:person@example.com" },
      { kind: "LINKEDIN", url: "javascript:alert(1)" },
    ],
    experiences: [
      {
        type: "CLIENT_ENGAGEMENT",
        organization: "Verified client",
        role: "Verified role",
        location: null,
        summary: "Verified summary.",
        startYear: 2024,
        startMonth: 6,
        endYear: null,
        endMonth: null,
        isCurrent: true,
      },
    ],
  });

  assert.equal(mapped.experience[0]?.period, "Jun 2024 — Present");
  assert.equal(mapped.experience[0]?.engagement, "Client engagement");
  assert.equal(mapped.links.email, "mailto:person@example.com");
  assert.equal(mapped.links.linkedIn, null);
});

