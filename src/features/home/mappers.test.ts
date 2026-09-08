import assert from "node:assert/strict";
import test from "node:test";

import { mapPublicHomeRecord, type PublicHomeRecord } from "./mappers";

const baseRecord: PublicHomeRecord = {
  name: "Verified person",
  positioning: ["AI/ML Engineer"],
  introduction: "Verified introduction.",
  location: null,
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

