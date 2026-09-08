import assert from "node:assert/strict";
import test from "node:test";

import {
  adminNestedResourceSchema,
  parseAdminNestedContentInput,
  projectNestedContentInputSchema,
  researchNestedContentInputSchema,
} from "./schemas";

const firstId = "11111111-1111-4111-8111-111111111111";
const secondId = "22222222-2222-4222-8222-222222222222";
const thirdId = "33333333-3333-4333-8333-333333333333";
const revision = "2026-08-22T17:30:00.000Z";

const validResearchInput = {
  revision,
  removedIds: [secondId],
  confirmRemoval: true as const,
  questions: [
    {
      id: firstId,
      key: "first-question",
      question: "A verified research question?",
    },
    {
      key: "new-question",
      question: "A new verified research question?",
    },
  ],
};

const validProjectInput = {
  revision,
  removedIds: [],
  contributions: [
    {
      id: firstId,
      key: "implementation",
      label: "Implementation",
      description: "Verified contribution.",
    },
  ],
  features: [
    {
      id: secondId,
      key: "progress-tracking",
      name: "Progress tracking",
      description: null,
    },
  ],
  technologies: [
    {
      id: thirdId,
      slug: "typescript",
      name: "TypeScript",
    },
  ],
};

function issuePaths(result: ReturnType<typeof researchNestedContentInputSchema.safeParse>) {
  return result.success
    ? []
    : result.error.issues.map((issue) => issue.path.join("."));
}

test("nested resource dispatch is a closed allowlist", () => {
  assert.equal(adminNestedResourceSchema.safeParse("research-projects").success, true);
  assert.equal(adminNestedResourceSchema.safeParse("projects").success, true);
  for (const resource of ["articles", "users", "../projects", "projects/edit"]) {
    assert.equal(adminNestedResourceSchema.safeParse(resource).success, false);
  }
});

test("research nested input accepts a strict full ordered-array payload", () => {
  const result = researchNestedContentInputSchema.safeParse(validResearchInput);
  assert.equal(result.success, true);
  if (result.success) {
    assert.deepEqual(result.data, validResearchInput);
  }

  assert.equal(
    researchNestedContentInputSchema.safeParse({
      revision,
      removedIds: [],
    }).success,
    false,
  );
  assert.equal(
    researchNestedContentInputSchema.safeParse({
      ...validResearchInput,
      parentId: firstId,
    }).success,
    false,
  );
  assert.equal(
    researchNestedContentInputSchema.safeParse({
      ...validResearchInput,
      questions: [
        {
          ...validResearchInput.questions[0],
          sortOrder: 99,
          profileId: secondId,
        },
      ],
    }).success,
    false,
  );
});

test("nested revisions require an ISO timestamp and confirmation is literal true", () => {
  for (const invalidRevision of [
    undefined,
    null,
    new Date(revision),
    "2026-08-22",
    "not-a-revision",
  ]) {
    assert.equal(
      researchNestedContentInputSchema.safeParse({
        ...validResearchInput,
        revision: invalidRevision,
      }).success,
      false,
    );
  }

  assert.equal(
    researchNestedContentInputSchema.safeParse({
      ...validResearchInput,
      confirmRemoval: undefined,
    }).success,
    true,
  );
  for (const confirmRemoval of [false, "true", 1, null]) {
    assert.equal(
      researchNestedContentInputSchema.safeParse({
        ...validResearchInput,
        confirmRemoval,
      }).success,
      false,
    );
  }
});

test("research questions reject duplicate identifiers, keys, and removed IDs at precise paths", () => {
  const duplicateId = researchNestedContentInputSchema.safeParse({
    ...validResearchInput,
    questions: validResearchInput.questions.map((question) => ({
      ...question,
      id: firstId,
    })),
  });
  assert.equal(duplicateId.success, false);
  assert.ok(issuePaths(duplicateId).includes("questions.1.id"));

  const duplicateKey = researchNestedContentInputSchema.safeParse({
    ...validResearchInput,
    questions: validResearchInput.questions.map((question) => ({
      ...question,
      key: "duplicate-key",
    })),
  });
  assert.equal(duplicateKey.success, false);
  assert.ok(issuePaths(duplicateKey).includes("questions.1.key"));

  const duplicateRemoval = researchNestedContentInputSchema.safeParse({
    ...validResearchInput,
    removedIds: [secondId, secondId],
  });
  assert.equal(duplicateRemoval.success, false);
  assert.ok(issuePaths(duplicateRemoval).includes("removedIds.1"));
});

test("project nested input requires every collection and rejects cross-section ID reuse", () => {
  assert.equal(projectNestedContentInputSchema.safeParse(validProjectInput).success, true);
  assert.equal(
    projectNestedContentInputSchema.safeParse({
      revision,
      removedIds: [],
      contributions: [],
      features: [],
    }).success,
    false,
  );

  const duplicateId = projectNestedContentInputSchema.safeParse({
    ...validProjectInput,
    features: [{ ...validProjectInput.features[0], id: firstId }],
  });
  assert.equal(duplicateId.success, false);
  if (!duplicateId.success) {
    assert.ok(
      duplicateId.error.issues
        .map((issue) => issue.path.join("."))
        .includes("features.0.id"),
    );
  }
});

test("project nested natural keys and slugs are unique and canonical", () => {
  assert.equal(
    projectNestedContentInputSchema.safeParse({
      ...validProjectInput,
      contributions: [
        validProjectInput.contributions[0],
        {
          key: validProjectInput.contributions[0].key,
          label: "Duplicate",
          description: "Duplicate key.",
        },
      ],
    }).success,
    false,
  );
  assert.equal(
    projectNestedContentInputSchema.safeParse({
      ...validProjectInput,
      technologies: [
        validProjectInput.technologies[0],
        { slug: "typescript", name: "Duplicate" },
      ],
    }).success,
    false,
  );
  assert.equal(
    projectNestedContentInputSchema.safeParse({
      ...validProjectInput,
      technologies: [{ slug: "Type Script", name: "Invalid slug" }],
    }).success,
    false,
  );
});

test("nested parser dispatches to the exact resource schema", () => {
  assert.deepEqual(
    parseAdminNestedContentInput("research-projects", validResearchInput),
    validResearchInput,
  );
  assert.deepEqual(
    parseAdminNestedContentInput("projects", validProjectInput),
    validProjectInput,
  );
});

test("nested collection limits keep interactive transactions operationally bounded", () => {
  const questions = Array.from({ length: 26 }, (_, index) => ({
    key: `question-${index}`,
    question: `Verified question ${index}?`,
  }));
  assert.equal(
    researchNestedContentInputSchema.safeParse({
      revision,
      removedIds: [],
      questions,
    }).success,
    false,
  );

  const technologies = Array.from({ length: 51 }, (_, index) => ({
    slug: `technology-${index}`,
    name: `Technology ${index}`,
  }));
  assert.equal(
    projectNestedContentInputSchema.safeParse({
      ...validProjectInput,
      technologies,
    }).success,
    false,
  );
});
