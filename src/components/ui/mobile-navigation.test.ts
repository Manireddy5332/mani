import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import type { KeyboardEvent, MouseEvent } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { AdminPageHeader } from "../admin/admin-page-header";
import { buttonStyles } from "./button";
import { MobileNavigation } from "./mobile-navigation";

const props = {
  label: "Menu",
  routeKey: "/research",
  children: createElement("a", { href: "/about" }, "About"),
};

test("mobile navigation renders a closed native disclosure with server-rendered links", () => {
  const html = renderToStaticMarkup(createElement(MobileNavigation, props));
  assert.match(html, /<details[^>]*>/);
  assert.doesNotMatch(html, /<details[^>]*\bopen(?:=|\s|>)/);
  assert.match(html, /<summary[^>]*>.*Menu.*<\/summary>/);
  assert.match(html, /<a href="\/about">About<\/a>/);
  assert.match(html, /min-h-11/);
  assert.match(html, /max-h-\[60svh\]/);
});

test("route changes remount the disclosure in its closed state", () => {
  assert.equal(MobileNavigation(props).key, "/research");
  assert.equal(MobileNavigation({ ...props, routeKey: "/about" }).key, "/about");
});

test("Escape closes the menu and returns keyboard focus to its summary", () => {
  let focused = false;
  const details = { open: true, querySelector: () => ({ focus: () => { focused = true; } }) };
  MobileNavigation(props).props.onKeyDown({
    key: "Escape", currentTarget: details,
  } as unknown as KeyboardEvent<HTMLDetailsElement>);
  assert.equal(details.open, false);
  assert.equal(focused, true);
});

test("non-Escape keys leave native disclosure keyboard behavior intact", () => {
  const details = { open: true, querySelector: () => assert.fail("Unexpected focus change") };
  MobileNavigation(props).props.onKeyDown({
    key: "Tab", currentTarget: details,
  } as unknown as KeyboardEvent<HTMLDetailsElement>);
  assert.equal(details.open, true);
});

test("following a link closes the menu without preventing navigation", () => {
  const details = { open: true };
  MobileNavigation(props).props.onClick({
    currentTarget: details, target: { closest: () => ({}) },
  } as unknown as MouseEvent<HTMLDetailsElement>);
  assert.equal(details.open, false);
});

test("clicking the summary does not interfere with its native toggle", () => {
  const details = { open: false };
  MobileNavigation(props).props.onClick({
    currentTarget: details, target: { closest: () => null },
  } as unknown as MouseEvent<HTMLDetailsElement>);
  assert.equal(details.open, false);
});

test("small buttons retain at least a 44px target with readable mobile text", () => {
  const classes = buttonStyles({ size: "sm" }).split(" ");
  assert.ok(classes.includes("min-h-11"));
  assert.ok(classes.includes("text-sm"));
  assert.ok(classes.includes("lg:text-xs"));
});

test("admin editor back links have a full touch target and retain their destination", () => {
  const html = renderToStaticMarkup(createElement(AdminPageHeader, {
    title: "Record", description: "Edit record", backHref: "/admin/projects",
  }));
  assert.match(html, /<a[^>]*class="[^"]*min-h-11[^"]*"[^>]*href="\/admin\/projects"/);
});
