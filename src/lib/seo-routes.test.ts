import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPublicSitemapPaths,
  privateRobotPaths,
  publicIndexPaths,
} from "./seo-routes";

test("sitemap paths include public indexes and eligible details once", () => {
  const paths = buildPublicSitemapPaths(
    ["/projects/verified-project", "/projects/verified-project"],
    ["/research/verified-direction"],
  );

  assert.deepEqual(paths.slice(0, publicIndexPaths.length), publicIndexPaths);
  assert.equal(paths.filter((path) => path === "/projects/verified-project").length, 1);
  assert.ok(paths.includes("/research/verified-direction"));
  assert.ok(paths.every((path) => !path.startsWith("/admin")));
});

test("robots exclusions cover every private application surface", () => {
  assert.deepEqual(privateRobotPaths, [
    "/admin",
    "/api/",
    "/sign-in",
    "/access-denied",
  ]);
});
