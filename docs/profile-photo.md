# Profile & About photo

One optional portrait is managed on the primary **Profile & About → Edit** screen.
Choose a JPEG, PNG, or WebP (up to 2 MiB and 20 megapixels), then **Save photo**.
Replacement is a separate save; **Remove photo** requires confirmation. These
operations do not resubmit profile text, publication state, or other CMS fields.
Only the published primary profile displays its selected photo on Home and About.

## Current Production status — verified 2026-09-27

The feature is **deployed and verified in Production** at
<https://manireddys-portfolio.vercel.app>. The owner uploaded and saved the current
portrait through Profile & About; Home and About display it successfully. The
approved hero placement, size, and responsive behavior are preserved.

## Storage configuration

The existing Vercel project uses separate private Blob stores:

- **Production:** `mani-profile-photo-prod`, connected to Production only, with
  Vercel-managed OIDC enabled.
- **Development:** `mani-profile-photo-dev`, connected to Development only.
- **Preview:** no Blob store connection; photo controls remain safely disabled.

The private `mani-profile-photo-dev` Blob store has already been created and
connected to **Development only** in the existing Vercel project. Do not recreate
it or connect it to Preview or Production. No store is created by the application
or its build. Future billing, environment connections, and deployments require
the normal approvals; the completed release does not authorize further changes.

The `@vercel/blob` SDK uses `BLOB_STORE_ID` with Vercel-managed OIDC on Vercel.
Runtime credentials are request-scoped and are not necessarily available as
`process.env.VERCEL_OIDC_TOKEN`. The application detects the store binding; the
SDK resolves and validates the runtime credentials. Never copy tokens into code
or pass them through the UI. Production does not require a long-lived read-write
token. A server-only `BLOB_READ_WRITE_TOKEN` for the Development store is supported
for local development.
The store may provision additional webhook variables, but this feature does not
use browser uploads or webhooks. No `NEXT_PUBLIC_*` storage variable is needed.

Local Development credentials are configured in the Git-ignored `.env.local`;
existing authentication and database settings were preserved. A real portrait has
been uploaded successfully using the isolated test environment described below.
Never paste credentials into chat or overwrite `.env.local` with an environment
pull. Future store connection/environment updates require a new approved Vercel
deployment to take effect.

In an environment without a storage binding, only the photo controls are disabled.
The rest of the existing portfolio and CMS remain usable. Production storage is
configured and its photo controls are working.

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

## Regression testing for future changes

Run `pnpm db:validate`, `pnpm typecheck`, `pnpm lint`, `pnpm test`, and `pnpm build`.
Unit tests use in-memory image fixtures and mocked storage/database operations;
they must not upload test portraits to production or alter existing records.

Before live mutation/regression testing, confirm that the running process's
`DATABASE_URL` and `DIRECT_URL` target an isolated non-production Neon database.
This was verified for the temporary `profile-photo-dev` branch and its
`portfolio_photo_test` database. That test branch's expiry was **2026-09-22 at
21:58 UTC**; it is not a current test target. Stop for approval and re-verify an
isolated environment before any further live mutation tests. The ignored local
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

## Historical local verification checkpoint — 2026-09-20

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
  it remains untracked and must stay out of future commits. `.env.local` was not changed.

## Completed Production release and verification

- [PR #5](https://github.com/Manireddy5332/mani/pull/5) delivered the feature.
  [PR #6](https://github.com/Manireddy5332/mani/pull/6) fixed storage detection to
  accept the OIDC store binding without requiring a build-time token in the
  runtime environment. SDK authorization remains unchanged.
- At the final audit, GitHub `main` and the READY Vercel Production deployment
  matched merge commit `efb39822a9cc0a5566d55621eef925577946d3ac`. Production uses
  the private Production-only store through OIDC; Development remains isolated
  and Preview remains disconnected.
- The owner completed the Production upload. Home and About serve the saved
  metadata-stripped WebP through the controlled, non-cacheable image route.
  Authenticated admin preview and **Reload photo controls** work, and the former
  storage-not-configured warning is absent.
- All **130 automated tests** and **43 live HTTP checks** passed. Unauthenticated
  admin routes redirect to sign-in; photo metadata and preview requests without
  a session are denied. Authorization, validation, replacement, removal, and
  cache/privacy safeguards remain covered by the automated suite and the earlier
  isolated live cycle. Destructive photo tests were not repeated in Production.
- Home/About rendering and responsive layouts passed. Academic CV bytes match
  the reviewed file, and research, project, experience, and other public routes
  remain available. Browser console/hydration checks were clear on the inspected
  Home and authenticated admin views.
- The read-only database inventory found one published primary profile linked to
  one private photo asset, the two existing migrations applied, and no unrelated
  content records updated since the feature merge. No schema change, migration,
  reseed, or Development-data transfer was needed for the feature.
- The final audit did not change Production content, storage configuration, or
  the current photo. No tracked secrets, temporary debug logging, or local test
  artifacts were found. The original local `Image.jpeg` was preserved untracked.

## Historical local release recheck — 2026-09-21

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
  requests were denied. Auth/sign-out/non-admin regression tests passed. Production
  verification was still pending at this historical checkpoint; the completed
  Production verification is recorded above.
- All 28 non-photo CMS checksums remained unchanged. The isolated database still
  contains one primary Profile linked to one MediaAsset. `.env.local` was unchanged.
- Secret scanning found no credential matches in the changed/new inputs or 40
  freshly built client artifacts. Credentials and generated directories are ignored.
- The read-only configuration inspection confirmed the owner had resolved the
  Preview-scope blocker. Runtime OIDC access and the Production upload had not yet
  been tested at this historical checkpoint; both are now verified as recorded
  above. Future Production photo changes still require separate approval.

## Workflow and approvals for future changes

1. Make scoped changes on a topic branch from current `main`, review the diff,
   and use the existing `Manireddy5332/mani` GitHub pull-request workflow. Exclude
   `Image.jpeg` and all local credentials/test artifacts from every commit.
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
5. Preserve the current owner-uploaded Production photo. Stop for separate approval
   before any further upload, replacement, removal, or other Production data change.
   Never automatically upload the Development test image or copy its media record.

No additional storage setup, dependency, schema change, or migration is required
for the deployed feature. If rollback is needed, use the normal approved rollback
process to restore a known-good application deployment; do not delete Blob objects
or edit database records as part of a code rollback.

## Released feature file manifest — PR #5

The original feature release contained the following 35 files. The three-file
OIDC detection fix in PR #6 modified files already in this list. `Image.jpeg` is
an untracked local test input and must stay out of commits. Never stage the workspace
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
