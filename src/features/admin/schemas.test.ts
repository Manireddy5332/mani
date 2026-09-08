import assert from "node:assert/strict";
import test from "node:test";

import {
  adminCreateInputSchemas,
  adminDeleteInputSchema,
  adminUpdateInputSchemas,
  adminVisibilitySchema,
} from "./schemas";

const validExperience = {
  slug: "test-experience",
  type: "EMPLOYMENT" as const,
  organization: "Test organization",
  role: "Test role",
  summary: "A test-only experience fixture.",
  startYear: 2022,
  startMonth: 2,
  endYear: 2023,
  endMonth: 3,
  isCurrent: false,
  featured: false,
  status: "DRAFT" as const,
  sortOrder: 0,
};

const validEducation = {
  slug: "test-education",
  institution: "Test institution",
  degree: "Test degree",
  startYear: 2020,
  endYear: 2022,
  isCurrent: false,
  gpa: 3.5,
  gpaScale: 4,
  status: "DRAFT" as const,
  sortOrder: 0,
};

const validCertification = {
  slug: "test-certification",
  name: "Test certification",
  issuer: "Test issuer",
  issueYear: 2023,
  issueMonth: 4,
  expiryYear: 2025,
  expiryMonth: 4,
  doesNotExpire: false,
  featured: false,
  status: "DRAFT" as const,
  sortOrder: 0,
};

const validPublication = {
  slug: "test-publication",
  title: "Test publication",
  authors: ["Test Author"],
  type: "CONFERENCE_PAPER" as const,
  stage: "WORKING_PAPER" as const,
  featured: false,
  status: "DRAFT" as const,
  sortOrder: 0,
};

function assertRejectedAt(
  result: { success: boolean; error?: { issues: Array<{ path: PropertyKey[] }> } },
  path: string,
) {
  assert.equal(result.success, false);
  assert.ok(
    result.error?.issues.some((issue) => issue.path[0] === path),
    `Expected a validation issue at ${path}.`,
  );
}

function assertRejectedUnknownField(
  result: {
    success: boolean;
    error?: {
      issues: Array<{
        path: PropertyKey[];
        keys?: string[];
      }>;
    };
  },
  field: string,
) {
  assert.equal(result.success, false);
  assert.ok(
    result.error?.issues.some(
      (issue) =>
        issue.path[0] === field || issue.keys?.includes(field) === true,
    ),
    `Expected ${field} to be rejected as an unknown field.`,
  );
}

test("period validation rejects incomplete, contradictory, and reversed dates", async (t) => {
  const invalidPeriods = [
    {
      name: "a start month without a start year",
      input: { ...validExperience, startYear: null },
      path: "startYear",
    },
    {
      name: "an end month without an end year",
      input: { ...validExperience, endYear: null },
      path: "endYear",
    },
    {
      name: "an end year without a start year",
      input: {
        ...validExperience,
        startYear: null,
        startMonth: null,
        endYear: 2023,
        endMonth: null,
      },
      path: "startYear",
    },
    {
      name: "an end year earlier than the start year",
      input: { ...validExperience, startYear: 2024, endYear: 2023 },
      path: "endYear",
    },
    {
      name: "an end month earlier in the same year",
      input: {
        ...validExperience,
        startYear: 2023,
        startMonth: 7,
        endYear: 2023,
        endMonth: 6,
      },
      path: "endYear",
    },
    {
      name: "an end date on a current record",
      input: { ...validExperience, isCurrent: true },
      path: "isCurrent",
    },
  ] as const;

  for (const invalidPeriod of invalidPeriods) {
    await t.test(invalidPeriod.name, () => {
      assertRejectedAt(
        adminCreateInputSchemas.experience.safeParse(invalidPeriod.input),
        invalidPeriod.path,
      );
    });
  }

  assert.equal(
    adminCreateInputSchemas.experience.safeParse(validExperience).success,
    true,
  );
  assert.equal(
    adminCreateInputSchemas.experience.safeParse({
      ...validExperience,
      endYear: null,
      endMonth: null,
      isCurrent: true,
    }).success,
    true,
  );
});

test("education validation keeps GPA within its declared scale", () => {
  assert.equal(
    adminCreateInputSchemas.education.safeParse(validEducation).success,
    true,
  );

  assertRejectedAt(
    adminCreateInputSchemas.education.safeParse({
      ...validEducation,
      gpa: 4.1,
      gpaScale: 4,
    }),
    "gpa",
  );
});

test("certification validation enforces coherent issue and expiry dates", async (t) => {
  const invalidCertifications = [
    {
      name: "issue month without issue year",
      input: { ...validCertification, issueYear: null },
      path: "issueYear",
    },
    {
      name: "expiry month without expiry year",
      input: { ...validCertification, expiryYear: null },
      path: "expiryYear",
    },
    {
      name: "expiry year without issue year",
      input: {
        ...validCertification,
        issueYear: null,
        issueMonth: null,
        expiryMonth: null,
      },
      path: "issueYear",
    },
    {
      name: "expiry before issue",
      input: { ...validCertification, expiryYear: 2022 },
      path: "expiryYear",
    },
    {
      name: "expiry month before issue month in the same year",
      input: {
        ...validCertification,
        issueYear: 2024,
        issueMonth: 6,
        expiryYear: 2024,
        expiryMonth: 5,
      },
      path: "expiryYear",
    },
    {
      name: "an expiry date on a non-expiring certification",
      input: { ...validCertification, doesNotExpire: true },
      path: "doesNotExpire",
    },
  ] as const;

  for (const invalidCertification of invalidCertifications) {
    await t.test(invalidCertification.name, () => {
      assertRejectedAt(
        adminCreateInputSchemas.certifications.safeParse(
          invalidCertification.input,
        ),
        invalidCertification.path,
      );
    });
  }

  assert.equal(
    adminCreateInputSchemas.certifications.safeParse({
      ...validCertification,
      expiryYear: null,
      expiryMonth: null,
      doesNotExpire: true,
    }).success,
    true,
  );
});

test("publications require at least one verified author and normalize author lists", () => {
  for (const authors of [[], "", "\n  \n"]) {
    assertRejectedAt(
      adminCreateInputSchemas.publications.safeParse({
        ...validPublication,
        authors,
      }),
      "authors",
    );
  }

  const result = adminCreateInputSchemas.publications.safeParse({
    ...validPublication,
    authors: " Test Author \nSecond Author\nTest Author",
  });
  assert.equal(result.success, true);
  if (result.success) {
    assert.deepEqual(result.data.authors, ["Test Author", "Second Author"]);
  }
});

test("strict input schemas reject server-owned fields", () => {
  const validProfile = {
    slug: "test-profile",
    name: "Test profile",
    positioning: ["Test positioning"],
    headline: "Test headline",
    introduction: "Test introduction",
    about: [],
    status: "DRAFT" as const,
  };
  const serverOwnedFields = [
    "id",
    "profileId",
    "source",
    "seedKey",
    "createdAt",
    "updatedAt",
    "deletedAt",
  ] as const;

  for (const field of serverOwnedFields) {
    assertRejectedUnknownField(
      adminCreateInputSchemas.profile.safeParse({
        ...validProfile,
        [field]: "forged-value",
      }),
      field,
    );
  }
});

test("updates require a complete strict record instead of accepting partial patches", () => {
  assert.equal(
    adminUpdateInputSchemas.experience.safeParse(validExperience).success,
    true,
  );
  assert.equal(
    adminUpdateInputSchemas.experience.safeParse({
      summary: "A forged partial update.",
    }).success,
    false,
  );
  assertRejectedUnknownField(
    adminUpdateInputSchemas.experience.safeParse({
      ...validExperience,
      profileId: "forged-profile-id",
    }),
    "profileId",
  );
});

test("social link protocols match their declared kind and reject credentials", () => {
  const base = {
    key: "professional-link",
    label: "Professional link",
    sortOrder: 0,
    status: "DRAFT" as const,
  };

  assert.equal(
    adminCreateInputSchemas["social-links"].safeParse({
      ...base,
      kind: "EMAIL",
      url: "mailto:person@example.com",
    }).success,
    true,
  );
  assertRejectedAt(
    adminCreateInputSchemas["social-links"].safeParse({
      ...base,
      kind: "EMAIL",
      url: "https://example.com/contact",
    }),
    "url",
  );
  assert.equal(
    adminCreateInputSchemas["social-links"].safeParse({
      ...base,
      kind: "GITHUB",
      url: "https://github.com/example",
    }).success,
    true,
  );
  for (const url of [
    "mailto:person@example.com",
    "ftp://example.com/profile",
    "https://user:password@example.com/profile",
  ]) {
    assertRejectedAt(
      adminCreateInputSchemas["social-links"].safeParse({
        ...base,
        kind: "GITHUB",
        url,
      }),
      "url",
    );
  }
});

test("visibility and ordering accept only known bounded values", () => {
  assert.equal(adminVisibilitySchema.safeParse("DRAFT").success, true);
  assert.equal(adminVisibilitySchema.safeParse("PUBLISHED").success, true);
  assert.equal(adminVisibilitySchema.safeParse("ARCHIVED").success, false);

  for (const status of ["DRAFT", "PUBLISHED"] as const) {
    const result = adminCreateInputSchemas.experience.safeParse({
      ...validExperience,
      status,
      sortOrder: "12",
    });
    assert.equal(result.success, true);
    if (result.success) {
      assert.equal(result.data.status, status);
      assert.equal(result.data.sortOrder, 12);
    }
  }

  for (const status of ["ARCHIVED", "PRIVATE", "published", "", null]) {
    assertRejectedAt(
      adminCreateInputSchemas.experience.safeParse({
        ...validExperience,
        status,
      }),
      "status",
    );
  }

  for (const sortOrder of [-1, 1.5, 1_000_001, "12px", Number.POSITIVE_INFINITY]) {
    assertRejectedAt(
      adminCreateInputSchemas.experience.safeParse({
        ...validExperience,
        sortOrder,
      }),
      "sortOrder",
    );
  }
});

test("delete confirmation accepts only the explicit confirmed payload", () => {
  assert.equal(adminDeleteInputSchema.safeParse({ confirmed: true }).success, true);
  assert.equal(
    adminDeleteInputSchema.safeParse({ confirmed: false }).success,
    false,
  );
  assertRejectedUnknownField(
    adminDeleteInputSchema.safeParse({
      confirmed: true,
      id: "forged-record-id",
    }),
    "id",
  );
});
