import assert from "node:assert/strict";
import test from "node:test";

import { mapPublicContactRecord, type PublicContactRecord } from "./mappers";

const baseRecord: PublicContactRecord = {
  name: "Verified person",
  location: null,
  socialLinks: [],
  projects: [],
};

test("contact mapper returns an honest empty method list", () => {
  assert.deepEqual(mapPublicContactRecord(baseRecord).methods, []);
});

test("contact mapper exposes only safe published link shapes", () => {
  const mapped = mapPublicContactRecord({
    ...baseRecord,
    location: "Verified location",
    socialLinks: [
      {
        key: "email",
        kind: "EMAIL",
        label: "Email",
        url: "mailto:person@example.com",
        handle: null,
      },
      {
        key: "unsafe",
        kind: "WEBSITE",
        label: "Unsafe",
        url: "javascript:alert(1)",
        handle: null,
      },
    ],
    projects: [
      {
        title: "Verified project",
        shortTitle: null,
        repositoryUrl: "https://example.com/repository",
      },
    ],
  });

  assert.deepEqual(
    mapped.methods.map((method) => method.kind),
    ["email", "location", "repository"],
  );
  assert.equal(mapped.methods[0]?.value, "person@example.com");
});

