import assert from "node:assert/strict";
import test from "node:test";

import {
  applicationOriginsMatch,
  isSecureApplicationOrigin,
} from "./origin-policy";

test("accepts HTTPS origins and local HTTP loopback origins", () => {
  assert.equal(isSecureApplicationOrigin("https://portfolio.example"), true);
  assert.equal(isSecureApplicationOrigin("https://portfolio.example:8443"), true);
  assert.equal(isSecureApplicationOrigin("http://localhost:3000"), true);
  assert.equal(isSecureApplicationOrigin("http://127.0.0.1:3000"), true);
  assert.equal(isSecureApplicationOrigin("http://[::1]:3000"), true);
});

test("rejects insecure remote and non-origin URLs", () => {
  assert.equal(isSecureApplicationOrigin("http://portfolio.example"), false);
  assert.equal(isSecureApplicationOrigin("https://portfolio.example/admin"), false);
  assert.equal(isSecureApplicationOrigin("https://portfolio.example?preview=1"), false);
  assert.equal(isSecureApplicationOrigin("https://user:pass@portfolio.example"), false);
  assert.equal(isSecureApplicationOrigin("ftp://portfolio.example"), false);
  assert.equal(isSecureApplicationOrigin(" https://portfolio.example"), false);
});

test("compares normalized origins without accepting a different host", () => {
  assert.equal(
    applicationOriginsMatch(
      "https://PORTFOLIO.example",
      "https://portfolio.example/",
    ),
    true,
  );
  assert.equal(
    applicationOriginsMatch(
      "https://portfolio.example",
      "https://www.portfolio.example",
    ),
    false,
  );
});
