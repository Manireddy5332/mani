export type PublicContentSource =
  | "database"
  | "last-known-good"
  | "static-fallback";

export type PublicContentResult<T> = Readonly<{
  data: T;
  source: PublicContentSource;
}>;

type StoredEntry = Readonly<{
  tags: ReadonlySet<string>;
  value: unknown;
}>;

export class LastKnownGoodStore {
  private readonly entries = new Map<string, StoredEntry>();

  constructor(private readonly maxEntries = 128) {}

  remember<T>(key: string, tags: readonly string[], value: T): void {
    this.entries.delete(key);
    this.entries.set(key, { tags: new Set(tags), value });
    while (this.entries.size > this.maxEntries) {
      const oldestKey = this.entries.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      this.entries.delete(oldestKey);
    }
  }

  read<T>(key: string): T | undefined {
    return this.entries.get(key)?.value as T | undefined;
  }

  clearByTags(tags: readonly string[]): void {
    if (tags.length === 0) return;
    const invalidated = new Set(tags);
    for (const [key, entry] of this.entries) {
      if ([...entry.tags].some((tag) => invalidated.has(tag))) {
        this.entries.delete(key);
      }
    }
  }

  clear(): void {
    this.entries.clear();
  }
}

type ResolvePublicContentOptions<T> = Readonly<{
  key: string;
  tags: readonly string[];
  load: () => Promise<T>;
  fallback: () => T;
  store: LastKnownGoodStore;
}>;

/**
 * A successful empty result is authoritative. Fallback content is used only
 * when the database/cache read throws, so archived content is never revived by
 * an empty published query.
 */
export async function resolvePublicContent<T>({
  key,
  tags,
  load,
  fallback,
  store,
}: ResolvePublicContentOptions<T>): Promise<PublicContentResult<T>> {
  try {
    const data = await load();
    store.remember(key, tags, data);
    return { data, source: "database" };
  } catch {
    const lastKnownGood = store.read<T>(key);
    if (lastKnownGood !== undefined) {
      return { data: lastKnownGood, source: "last-known-good" };
    }
    return { data: fallback(), source: "static-fallback" };
  }
}
