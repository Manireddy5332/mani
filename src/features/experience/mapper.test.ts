import assert from "node:assert/strict";
import test from "node:test";

import { mapExperiencePageContent } from "./mapper";
import type { ExperiencePageSource } from "./types";

test("maps published-source shapes without fabricating date precision", () => {
  const source: ExperiencePageSource = {
    experiences: [
      {
        type: "CLIENT_ENGAGEMENT",
        organization: "Verified organization",
        role: "Verified role",
        location: null,
        summary: "Verified summary",
        practiceAreas: ["Machine learning"],
        startYear: 2024,
        startMonth: 6,
        endYear: null,
        endMonth: null,
        isCurrent: true,
      },
      {
        type: "EMPLOYMENT",
        organization: "Earlier organization",
        role: "Earlier role",
        location: "Verified location",
        summary: "Earlier summary",
        practiceAreas: [],
        startYear: 2020,
        startMonth: null,
        endYear: 2022,
        endMonth: null,
        isCurrent: false,
      },
    ],
    education: [
      {
        degree: "Verified degree",
        institution: "Verified institution",
        startYear: 2022,
        startMonth: 8,
        endYear: 2024,
        endMonth: 5,
        isCurrent: false,
      },
      {
        degree: "Undated degree",
        institution: "Undated institution",
        startYear: null,
        startMonth: null,
        endYear: null,
        endMonth: null,
        isCurrent: false,
      },
    ],
    skillCategories: [
      { name: "Published category", skills: [{ name: "Published skill" }] },
      { name: "Empty category", skills: [] },
    ],
  };

  const result = mapExperiencePageContent(source);

  assert.equal(result.engagements[0]?.engagement, "Client engagement");
  assert.equal(result.engagements[0]?.period, "Jun 2024 — Present");
  assert.equal(result.engagements[0]?.location, null);
  assert.equal(result.engagements[1]?.engagement, "Employment");
  assert.equal(result.engagements[1]?.period, "2020 — 2022");
  assert.equal(result.education[0]?.period, "Aug 2022 — May 2024");
  assert.equal(result.education[1]?.period, null);
  assert.deepEqual(result.expertise, [
    { category: "Published category", skills: ["Published skill"] },
  ]);
  assert.ok(result.practiceMap.length > 0);
});

test("keeps an authoritative empty database result empty", () => {
  const result = mapExperiencePageContent({
    experiences: [],
    education: [],
    skillCategories: [],
  });

  assert.deepEqual(result, {
    engagements: [],
    education: [],
    expertise: [],
    practiceMap: [],
  });
});

test("maps every supported experience type to visitor-facing language", () => {
  const types = [
    ["CLIENT_ENGAGEMENT", "Client engagement"],
    ["EMPLOYMENT", "Employment"],
    ["INTERNSHIP", "Internship"],
    ["CONTRACT", "Contract"],
    ["OTHER", "Professional experience"],
  ] as const;

  for (const [type, expected] of types) {
    const result = mapExperiencePageContent({
      experiences: [
        {
          type,
          organization: "Organization",
          role: "Role",
          location: null,
          summary: "Summary",
          practiceAreas: [],
          startYear: null,
          startMonth: null,
          endYear: null,
          endMonth: null,
          isCurrent: false,
        },
      ],
      education: [],
      skillCategories: [],
    });

    assert.equal(result.engagements[0]?.engagement, expected);
    assert.equal(result.engagements[0]?.period, null);
  }
});
