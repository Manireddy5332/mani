const PUBLIC_SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const MAX_PUBLIC_SLUG_LENGTH = 120;

export function isPublicResearchSlug(value: string): boolean {
  return (
    value.length > 0 &&
    value.length <= MAX_PUBLIC_SLUG_LENGTH &&
    PUBLIC_SLUG_PATTERN.test(value)
  );
}
