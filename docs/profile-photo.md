# Profile & About photo

One optional portrait is managed on the primary **Profile & About → Edit** screen.
Choose a JPEG, PNG, or WebP (up to 2 MiB and 20 megapixels), then **Save photo**.
Replacement is a separate save; **Remove photo** requires confirmation. These
operations do not resubmit profile text, publication state, or other CMS fields.
Only the published primary profile displays its selected photo on Home and About.

## Development storage setup

The private `mani-profile-photo-dev` Blob store has already been created and
connected to **Development only** in the existing Vercel project. Do not recreate
it or connect it to Preview or Production. No store is created by the application
or its build. Billing, further environment connections, and deployment each require
separate approval.

The `@vercel/blob` SDK uses `BLOB_STORE_ID` with Vercel-managed
`VERCEL_OIDC_TOKEN` on Vercel. Let the SDK obtain and refresh the token; never pass
it through the UI or manually copy it into code. A server-only
`BLOB_READ_WRITE_TOKEN` for that private store is supported for local development.
The store may provision additional webhook variables, but this feature does not
use browser uploads or webhooks. No `NEXT_PUBLIC_*` storage variable is needed.

Local Development credentials are configured in the Git-ignored `.env.local`;
existing authentication and database settings were preserved. A real portrait has
been uploaded successfully using the isolated test environment below. Never paste credentials into chat or
overwrite `.env.local` with an environment pull. A store connection/environment
update takes effect in a new Vercel deployment; no deployment is approved yet.

Until storage credentials are available, only the photo controls are disabled.
The rest of the existing portfolio and CMS remain usable.

## Implementation and privacy

- No Prisma migration: reuse `Profile.avatarId` and `MediaAsset`.
- Server-side administrator checks use the existing authorization guard. Mutations
  additionally enforce the configured same origin and removal confirmation.
- Validate actual decoded format, byte/pixel limits, and reject animated files.
  Normalize orientation, strip embedded EXIF/GPS metadata, resize within 960px,
  and encode WebP in memory. Nothing is uploaded to an ephemeral filesystem.
- Private blobs use immutable, server-generated keys. PostgreSQL changes only the
  avatar pointer, its automatic timestamp, and the feature's media metadata.
- An optimistic pointer check prevents a stale tab from overwriting another photo
  save. A database transaction creates media metadata and swaps the pointer together.
- After save/removal, expire the existing profile-related public cache tags and
  paths and clear the process-local last-known-good entries. Static fallback
  remains unchanged, with no photo.
- Public image requests re-check the current published avatar in PostgreSQL;
  admin previews re-check authorization. No storage URL/key is sent to the UI.
  Image responses are `no-store`; the portrait bypasses Next's shared optimizer
  because it is already optimized at upload time. Missing/failed images collapse
  to the existing no-photo public layout.
- The Next image optimizer allows only bundled static media; a direct optimizer
  request cannot cache either the public photo route or authenticated preview.
- Delete only detached, feature-owned media that is not referenced by any other
  CMS relation. A cleanup failure cannot undo a successful photo save. Storage and
  PostgreSQL cannot share a transaction: an uncertain database commit or failed
  storage deletion may leave an unreferenced private blob for later manual cleanup.
  Never automatically delete a blob that might still be selected.

## Verification before release

Run `pnpm db:validate`, `pnpm typecheck`, `pnpm lint`, `pnpm test`, and `pnpm build`.
Unit tests use in-memory image fixtures and mocked storage/database operations;
they must not upload test portraits to production or alter existing records.

Before live sign-in or photo-save testing, confirm that the running process's
`DATABASE_URL` and `DIRECT_URL` target an isolated non-production Neon database.
This was verified for the temporary `profile-photo-dev` branch and its
`portfolio_photo_test` database. The branch expires on **2026-09-22 at 21:58 UTC**;
never use it after expiry without re-verifying the environment. The ignored local
test harness overrides both connections in memory and leaves `.env.local`
unchanged. Do not start ordinary `pnpm dev` for this isolated test, because the
existing `.env.local` database settings are not the test target.
A Development-only Blob store does **not**
isolate PostgreSQL writes: the existing repository still saves `Profile.avatarId`,
its automatic timestamp, and `MediaAsset` metadata to the configured database.
Do not run these live tests against existing production records. If an isolated
database is unavailable, stop for the owner's configuration/approval; do not create
one, migrate, or seed automatically.

Once database isolation is confirmed, use an approved non-production environment
to verify upload → private Blob → avatar metadata → public Home/About, replace,
remove, stale-tab conflict, unpublished-profile image denial, and storage failure
fallback. Also verify signed-out and non-admin upload/preview requests are denied.
Do not upload a placeholder portrait or modify existing production content to test
this feature without explicit approval.

## Verification checkpoint — 2026-09-20

- Prisma validation and generation, lint, TypeScript, all 127 automated tests,
  and the local production build passed. The build used the isolated test database.
- All 16 local route/security smoke checks passed, including signed-out admin
  redirects, unauthorized photo GET/PUT/DELETE denial, unknown-photo 404s, and
  rejection of shared-image-optimizer access to protected photo routes.
- Authenticated admin photo controls and the previously uploaded portrait work.
  Home and About both deliver the optimized WebP with `no-store` headers.
- Home passed 390px mobile, 768px tablet, and 1280px desktop layout checks;
  About's portrait loaded correctly on a narrow viewport with no horizontal overflow.
  The approved desktop hero layout has not been altered during this verification.
- Automated tests cover replacement/removal ordering, validation, non-admin denial,
  stale revisions, storage failures, safe error responses, and cache/privacy rules.
- The isolated live replace → remove → restore cycle passed. The owner confirmed
  the browser removal dialog manually; restoration used the supplied `Image.jpeg`.
  Home and About serve image bytes identical to the approved pre-test portrait.
  Removed image URLs return 404 and the no-photo fallback was verified on both pages.
- Final isolated state: one primary Profile linked to one MediaAsset; only the
  restored object remains in this test profile's Development Blob prefix.
  Checksums of all 28 non-photo CMS tables/projections match the pre-test baseline
  (excluding only the profile avatar pointer and its automatic update timestamp).
- Final browser checks used the successful production build running **locally**
  against the isolated database and Development storage after the long-running
  development preview stalled. The separate runtime's existing 300-second content
  cache revalidated; final Home/About image and route checks passed. No application
  code or cache-policy change was made. Console/hydration checks were clear.
- The original JPEG is a local restore/test input, not a bundled public asset;
  keep it out of the eventual commit. `.env.local` was not changed.

## Production release gates

1. The isolated live cycle and local release checks are complete. Repeat relevant
   checks if implementation changes before release; do not alter the approved layout.
2. The owner created the separate private `mani-profile-photo-prod` store and
   corrected its environment scope. The final read-only Vercel inspection on
   2026-09-21 confirms that its active project connection and `BLOB_STORE_ID` are
   **Production only**. The private `mani-profile-photo-dev` store and its
   `BLOB_READ_WRITE_TOKEN` remain **Development only**; Preview has neither Blob
   connection. The prior Preview-scope blocker is resolved. No storage settings
   were changed by the agent. Stop before billing, upgrades, or account actions.
3. Project OIDC is enabled and Production has `BLOB_STORE_ID`; Vercel manages the
   short-lived runtime token. No Production read-write token is required by this
   implementation. These are configuration checks, not a deployed photo round trip.
   Keep storage credentials in Vercel's protected environment configuration only.
   Never copy local test database credentials into Vercel.
   No Prisma schema change, migration, reseed, or test-data copy is needed.
4. Review and approve the feature branch/PR and deployment separately. A push may
   create a Preview deployment; it must not silently connect Development storage
   to Preview or Production. Photo controls safely remain disabled without storage.
5. After the approved production release, the owner uploads the intended portrait
   through Profile & About. Do not automatically copy the isolated test avatar,
   media metadata, or other CMS data into production. Verify public rendering,
   signed-out protection, and photo-only updates without editing existing text.

Current status: **not deployed**. The agent did not change Production configuration
or data and performed only read-only inspection of the owner's storage setup. Local feature
verification and storage-configuration checks are complete. Branch/PR review,
deployment approval, and post-deployment verification remain release gates. No development photo was
transferred to Production.
The ordinary application rollback process remains applicable;
do not delete storage objects or database records as part of a code rollback.

## Final release recheck — 2026-09-21

- The existing implementation was preserved without further application or layout
  edits. Only this documentation was updated during the release recheck.
- Prisma validation/generation, lint, standalone TypeScript, 127/127 tests, and the
  production build passed again. No migrations, seeds, or deployments were run.
- The build ran locally against the isolated test database and Development storage.
  All 16 route/security smoke checks and five additional detail-route, unknown-slug
  404, and Academic CV checks passed. The Academic CV returned a valid PDF.
- Home and About serve the exact approved restored image, with no-store headers;
  both previous photo URLs remain inaccessible. The previously completed live
  upload/replace/remove/restore cycle was not repeated or copied to Production.
- Home and About passed 390px, 768px, and 1280px responsive checks. The desktop
  portrait remains beside the introduction at the approved approximately 271px
  displayed width. Browser console error/warning checks were clear.
- The existing authenticated local admin session loaded the photo controls and
  current protected image successfully; no save/removal was performed. Signed-out
  requests were denied. Auth/sign-out/non-admin regression tests passed; a fresh
  production OAuth/photo test remains deferred until an approved release.
- All 28 non-photo CMS checksums remained unchanged. The isolated database still
  contains one primary Profile linked to one MediaAsset. `.env.local` was unchanged.
- Secret scanning found no credential matches in the changed/new inputs or 40
  freshly built client artifacts. Credentials and generated directories are ignored.
- The final read-only Production configuration inspection confirms the owner
  resolved the Preview-scope blocker. Runtime OIDC access on Vercel cannot be
  certified until an approved deployment; no Production Blob upload or photo-data
  transfer was tested. Ask for separate approval before any production photo save.

## Deployment sequence and approvals

1. Obtain approval to commit the 35 feature files below on `profile-photo-support`,
   push that branch to the existing `Manireddy5332/mani` origin, and create a pull
   request into `main`. Exclude `Image.jpeg` and all local credentials/test artifacts.
2. Let the normal GitHub/Vercel Preview checks run and review the exact diff. Do not
   attach either Blob store to Preview to enable photo testing there: its photo
   controls should remain safely disabled. Do not change Preview database settings
   or perform content writes as part of this review.
3. Obtain separate approval to merge into `main`. Let the existing Vercel workflow
   deploy the approved commit; do not run migrations, seed, or transfer test records.
4. Verify the production deployment and public routes, Academic CV, metadata,
   server-side admin protection, and sign-in/sign-out. Google account interaction
   may require the owner. Inspect the production photo controls without saving.
   Confirm the new runtime receives its Production-only Blob/OIDC configuration;
   stop rather than printing credentials or changing authentication if it fails.
5. Stop for separate approval before any production photo upload, replacement,
   removal, or other production data change. Only after that approval should the
   owner-selected portrait be saved and its public Home/About rendering checked.
   Never automatically upload the Development test image or copy its media record.

No additional storage setup, new dependency, schema change, or migration is planned.
The configuration is ready for this approval-gated release, not proof of a completed
production photo transaction. If rollback is needed, restore the previous known-good
application deployment; do not delete Blob objects or edit database records.

## Exact feature file manifest

The proposed release contains 35 files. `Image.jpeg` is an untracked local test
input and must be excluded from the eventual commit. Never stage the workspace
indiscriminately; credentials, `.cache`, `.vercel`, and build outputs stay excluded.

Modified (17):

```text
.env.example
next.config.ts
package.json
pnpm-lock.yaml
src/app/admin/[resource]/[id]/edit/page.tsx
src/components/site/personal-portfolio-home.tsx
src/features/home/data.ts
src/features/home/mappers.test.ts
src/features/home/mappers.ts
src/features/home/repository.server.ts
src/features/home/types.ts
src/features/profile/components/about-page.tsx
src/features/profile/data.ts
src/features/profile/mappers.test.ts
src/features/profile/mappers.ts
src/features/profile/repository.server.ts
src/features/profile/types.ts
```

New (18):

```text
docs/profile-photo.md
src/app/api/admin/profile-photo/route.ts
src/app/profile-photo/[id]/route.ts
src/components/admin/admin-profile-photo.tsx
src/components/site/profile-photo.tsx
src/features/profile-photo/cache.test.ts
src/features/profile-photo/flow.test.ts
src/features/profile-photo/flow.ts
src/features/profile-photo/http.test.ts
src/features/profile-photo/http.ts
src/features/profile-photo/image.test.ts
src/features/profile-photo/image.ts
src/features/profile-photo/policy.test.ts
src/features/profile-photo/policy.ts
src/features/profile-photo/repository.server.ts
src/features/profile-photo/service.server.ts
src/features/profile-photo/storage.server.ts
src/features/profile-photo/types.ts
```
