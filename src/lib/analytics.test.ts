import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { Analytics } from "@vercel/analytics/next";

import { PublicAnalytics } from "../components/site/public-analytics";
import { beforeSendPublicPageview } from "./analytics";
import { publicIndexPaths } from "./seo-routes";

const origin = "https://portfolio.example";
const pageview = (path: string) => ({ type: "pageview" as const, url: `${origin}${path}` });

test("all existing public index pages are eligible for page views", () => {
  for (const path of publicIndexPaths) {
    assert.deepEqual(beforeSendPublicPageview(pageview(path)), pageview(path));
  }
});

test("public research and project detail URLs are eligible", () => {
  for (const path of ["/research/verified-research", "/projects/verified-project-2"]) {
    assert.deepEqual(beforeSendPublicPageview(pageview(path)), pageview(path));
  }
});

test("private, authentication, API and unknown routes fail closed", () => {
  for (const path of [
    "/admin", "/admin/", "/admin/profile", "/admin/profile/record/edit",
    "/sign-in", "/sign-in/", "/access-denied", "/api", "/api/auth/callback/google",
    "/api/profile-photo", "/auth/callback", "/account", "/private", "/dashboard",
    "/research/project/edit", "/projects/example/preview", "/about/private",
    "/writing/unreleased-article", "/admin?callbackURL=/", "/unknown",
  ]) {
    assert.equal(beforeSendPublicPageview(pageview(path)), null, path);
  }
});

test("public URLs omit all query parameters and fragments without mutating the event", () => {
  const original = pageview("/about/?email=example%40example.test&code=test-code#private-fragment");
  const copy = { ...original };
  assert.deepEqual(beforeSendPublicPageview(original), pageview("/about"));
  assert.deepEqual(original, copy);
});

test("malformed, credential-bearing and non-web URLs are rejected", () => {
  for (const url of [
    "not a URL", "/about", "javascript:alert(1)", "file:///about",
    "https://example-user:example-password@portfolio.example/about",
    `${origin}/projects/person%40example.test`, `${origin}/%61dmin`,
  ]) {
    assert.equal(beforeSendPublicPageview({ type: "pageview", url }), null);
  }
});

test("custom events are not collected in Phase 1", () => {
  assert.equal(beforeSendPublicPageview({
    type: "event", url: origin,
  }), null);
});

test("client-side public to private to public navigation keeps the filter active", () => {
  assert.deepEqual(
    ["/", "/admin", "/admin/projects", "/sign-in", "/about"].map(
      (path) => beforeSendPublicPageview(pageview(path)),
    ),
    [pageview("/"), null, null, null, pageview("/about")],
  );
});

test("the wrapper uses the official Next.js component and privacy filter", () => {
  const element = PublicAnalytics({ enabled: true });
  assert.ok(element);
  assert.equal(element.type, Analytics);
  assert.equal(element.props.beforeSend, beforeSendPublicPageview);
  assert.equal(element.props.debug, false);
});

test("tracking is absent when not enabled for Vercel Production", () => {
  assert.equal(PublicAnalytics({ enabled: false }), null);
});

test("the root layout mounts analytics once and only enables it in Vercel Production", () => {
  const layout = readFileSync(new URL("../app/layout.tsx", import.meta.url), "utf8");
  assert.equal(layout.match(/<PublicAnalytics\b/g)?.length, 1);
  assert.match(layout, /<PublicAnalytics enabled=\{process\.env\.VERCEL_ENV === "production"\} \/>/);
  assert.doesNotMatch(layout, /["']use client["']/);
});
