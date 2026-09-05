import assert from "node:assert/strict";
import test from "node:test";

import { mapWritingOverview } from "./mapper";

test("maps the eligible research direction and questions without articles", () => {
  const result = mapWritingOverview({
    sourceDirection: {
      title: "Verified research direction",
      stage: "EARLY_STAGE",
      format: "Verified review format",
      summary: "Verified summary",
      questions: [
        { question: "First verified question?" },
        { question: "Second verified question?" },
      ],
    },
  });

  assert.deepEqual(result, {
    sourceDirection: {
      title: "Verified research direction",
      stage: "Early-stage",
      format: "Verified review format",
      summary: "Verified summary",
      questions: [
        "First verified question?",
        "Second verified question?",
      ],
    },
    entries: [],
  });
});

test("preserves an absent current research direction as an empty state", () => {
  assert.deepEqual(mapWritingOverview({ sourceDirection: null }), {
    sourceDirection: null,
    entries: [],
  });
});

test("does not invent a missing research format or questions", () => {
  const result = mapWritingOverview({
    sourceDirection: {
      title: "Verified direction",
      stage: "IN_PROGRESS",
      format: null,
      summary: "Verified summary",
      questions: [],
    },
  });

  assert.equal(result.sourceDirection?.stage, "In progress");
  assert.equal(result.sourceDirection?.format, null);
  assert.deepEqual(result.sourceDirection?.questions, []);
  assert.deepEqual(result.entries, []);
});
