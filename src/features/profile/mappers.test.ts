import assert from "node:assert/strict";
import test from "node:test";

import {
  mapPublicProfileRecord,
  mapPublicResumeRecord,
  type PublicProfileRecord,
  type PublicResumeProfileRecord,
} from "./mappers";

function profileRecord(
  overrides: Partial<PublicProfileRecord> = {},
): PublicProfileRecord {
  return {
    name: "Verified Person",
    positioning: ["AI/ML Engineer"],
    headline: "Verified headline",
    introduction: "Verified introduction",
    about: ["Verified profile paragraph."],
    location: null,
    experiences: [],
    educationRecords: [],
    researchInterests: [],
    researchProjects: [],
    skillCategories: [],
    certifications: [],
    ...overrides,
  };
}

test("maps published profile rows without inventing missing optional content", () => {
  const result = mapPublicProfileRecord(
    profileRecord({
      experiences: [
        {
          type: "CLIENT_ENGAGEMENT",
          organization: "Verified organization",
          role: "Verified role",
          location: null,
          summary: "Verified summary",
          startYear: 2024,
          startMonth: 6,
          endYear: null,
          endMonth: null,
          isCurrent: true,
        },
      ],
      educationRecords: [
        {
          institution: "Verified university",
          degree: "Verified degree",
          startYear: 2022,
          startMonth: null,
          endYear: 2024,
          endMonth: null,
          isCurrent: false,
        },
      ],
      researchInterests: [{ name: "Verified interest" }],
      researchProjects: [
        {
          slug: "verified-research",
          title: "Verified research",
          summary: "Verified research summary",
          format: null,
          stage: "EARLY_STAGE",
          questions: [{ question: "Verified question?" }],
        },
      ],
      skillCategories: [
        { name: "Verified category", skills: [{ name: "Verified skill" }] },
        { name: "Empty category", skills: [] },
      ],
      certifications: [
        {
          name: "Verified certification",
          issuer: "Verified issuer",
          credentialUrl: "https://credentials.example/verified",
          issueYear: 2025,
          issueMonth: 1,
          expiryYear: null,
          expiryMonth: null,
          doesNotExpire: true,
        },
      ],
    }),
  );

  assert.equal(result.location, null);
  assert.deepEqual(result.experience[0], {
    engagement: "Client engagement",
    organization: "Verified organization",
    location: null,
    role: "Verified role",
    period: "Jun 2024 — Present",
    summary: "Verified summary",
    isCurrent: true,
  });
  assert.equal(result.education[0]?.period, "2022 — 2024");
  assert.equal(result.research?.format, null);
  assert.deepEqual(result.research?.questions, ["Verified question?"]);
  assert.deepEqual(result.expertise, [
    { category: "Verified category", skills: ["Verified skill"] },
  ]);
  assert.deepEqual(result.certifications, [
    {
      name: "Verified certification",
      issuer: "Verified issuer",
      period: "Jan 2025 — Does not expire",
      credentialUrl: "https://credentials.example/verified",
    },
  ]);
});

test("keeps successful empty profile relations empty", () => {
  const result = mapPublicProfileRecord(profileRecord());

  assert.deepEqual(result.experience, []);
  assert.deepEqual(result.education, []);
  assert.equal(result.research, null);
  assert.deepEqual(result.expertise, []);
  assert.deepEqual(result.certifications, []);
});

test("maps only safe public resume and professional-link URLs", () => {
  const record: PublicResumeProfileRecord = {
    ...profileRecord(),
    socialLinks: [
      { kind: "EMAIL", url: "mailto:verified@example.com" },
      { kind: "LINKEDIN", url: "https://www.linkedin.com/in/verified" },
    ],
    projects: [
      {
        slug: "verified-project",
        title: "Verified project",
        type: null,
        institution: null,
        advisor: null,
        role: null,
        summary: "Verified project summary",
        repositoryUrl: "javascript:alert(1)",
        technologies: [],
      },
    ],
    resumes: [
      {
        fileAsset: {
          publicUrl: "https://portfolio.example/cv.pdf",
          mimeType: "application/pdf",
        },
      },
    ],
  };

  const result = mapPublicResumeRecord(record);

  assert.equal(result.emailAddress, "verified@example.com");
  assert.match(result.cvRequestUrl ?? "", /^mailto:verified@example\.com\?/);
  assert.equal(result.linkedInUrl, "https://www.linkedin.com/in/verified");
  assert.equal(result.cvDownloadUrl, "https://portfolio.example/cv.pdf");
  assert.equal(result.project?.repositoryUrl, null);
  assert.equal(result.project?.type, null);
});

test("does not expose an unsupported resume asset", () => {
  const result = mapPublicResumeRecord({
    ...profileRecord(),
    socialLinks: [],
    projects: [],
    resumes: [
      {
        fileAsset: {
          publicUrl: "https://portfolio.example/archive.zip",
          mimeType: "application/zip",
        },
      },
    ],
  });

  assert.equal(result.cvDownloadUrl, null);
  assert.equal(result.cvRequestUrl, null);
  assert.equal(result.project, null);
});
