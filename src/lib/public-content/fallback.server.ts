import "server-only";

import {
  LastKnownGoodStore,
  resolvePublicContent,
  type PublicContentResult,
} from "@/lib/public-content/fallback";

const globalForPublicContent = globalThis as typeof globalThis & {
  portfolioPublicLastKnownGood?: LastKnownGoodStore;
};

function getStore(): LastKnownGoodStore {
  globalForPublicContent.portfolioPublicLastKnownGood ??=
    new LastKnownGoodStore();
  return globalForPublicContent.portfolioPublicLastKnownGood;
}

export async function loadPublicContent<T>(options: Readonly<{
  key: string;
  tags: readonly string[];
  load: () => Promise<T>;
  fallback: () => T;
}>): Promise<PublicContentResult<T>> {
  return resolvePublicContent({ ...options, store: getStore() });
}

export function clearPublicLastKnownGood(tags: readonly string[]): void {
  getStore().clearByTags(tags);
}

