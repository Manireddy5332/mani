# Production deployment and operations

This guide records the live deployment and provides the repeatable release,
verification, and rollback process. Never paste real values into source
control, build logs, issue trackers, or deployment notes.

## Current production deployment

| Setting | Active configuration |
| --- | --- |
| Platform | Vercel |
| Project | `manikanta-ai-portfolio` |
| Public brand | `Mani Reddy’s Portfolio` |
| Production URL | `https://manireddys-portfolio.vercel.app` |
| Transition hostname | `https://manikanta-ai-portfolio-pi.vercel.app` (retained until the new hostname is fully verified) |
| Framework | Next.js |
| Node.js | `24.x` |
| Package manager | `pnpm@11.16.0` through Corepack |
| Build command | `pnpm build` |
| Output | Next.js framework default |
| Function region | `iad1` |
| Database | Existing Neon PostgreSQL database |

The Vercel project is linked to GitHub repository `Manireddy5332/mani`, with
`main` configured as the production branch. The original Phase 10 release was
uploaded from the reviewed local workspace with Vercel CLI while source control
was being initialized; GitHub is now the authoritative release source.

For each change, create or update a topic branch, push it to GitHub, and open a
pull request into `main`. Review the complete diff and wait for the required
GitHub checks and Vercel Preview deployment to succeed. Merge only after those
checks are green. Vercel then creates the production deployment from the merged
`main` commit. Do not push unreviewed changes directly to `main` or promote a
feature-branch Preview deployment to production.

## Recommended target

The deployment uses Vercel's Next.js framework preset with the existing
repository root and package manager:

- Node.js: `24.x`
- Install command: `pnpm install --frozen-lockfile`
- Build command: `pnpm build`
- Output directory: framework default; do not override `.next`
- Runtime: Node.js, not Edge, for Better Auth, Prisma, and `pg`
- Build environment: `ENABLE_EXPERIMENTAL_COREPACK=1` so Vercel honors the
  pinned `pnpm@11.16.0`; verify the pinned install in a preview build
- Function region: `iad1`

The project does not require `vercel.json` for this configuration. Keep
Preview and Production environment-variable scopes separate. Do not enable
administrator OAuth on ephemeral preview URLs unless a stable preview domain,
separate OAuth configuration, and an isolated database branch have been
deliberately configured.

## Production environment contract

Configure these values in the hosting provider's encrypted environment store.
Production URLs must be HTTPS origins without paths, credentials, queries, or
fragments.

| Name | Production requirement |
| --- | --- |
| `NEXT_PUBLIC_SITE_URL` | Canonical public HTTPS origin, such as `https://portfolio.example`. This is intentionally public and is embedded at build time. |
| `BETTER_AUTH_URL` | The exact same HTTPS origin as `NEXT_PUBLIC_SITE_URL`. |
| `BETTER_AUTH_SECRET` | A new high-entropy production secret of at least 32 characters. Do not reuse the local value. |
| `GOOGLE_CLIENT_ID` | Client ID for a Google OAuth Web application authorized for the production origin. |
| `GOOGLE_CLIENT_SECRET` | Matching server-only Google OAuth client secret. |
| `ADMIN_EMAIL` | The exact verified Google identity permitted to administer the portfolio. |
| `DATABASE_URL` | Neon pooled runtime URL; the endpoint hostname contains `-pooler`. Retain Neon TLS parameters. |
| `ENABLE_EXPERIMENTAL_COREPACK` | Set to `1` so Vercel honors the pinned pnpm release. |
| `DIRECT_URL` | Matching Neon direct/non-pooled URL for controlled Prisma migration commands only. Deliberately absent from Vercel request-serving functions. |
| `SHADOW_DATABASE_URL` | Leave unset in production. Use only with a separate disposable database/branch while authoring migrations. |

Only `NEXT_PUBLIC_SITE_URL` is permitted in browser bundles. Every other value
above remains server-only.

## Google OAuth production setup

The production Google OAuth Web client is configured with both production
origins during the verified domain transition:

- authorized JavaScript origin:
  `https://manireddys-portfolio.vercel.app`
- authorized redirect URI:
  `https://manireddys-portfolio.vercel.app/api/auth/callback/google`
- transition authorized JavaScript origin:
  `https://manikanta-ai-portfolio-pi.vercel.app`
- transition authorized redirect URI:
  `https://manikanta-ai-portfolio-pi.vercel.app/api/auth/callback/google`

The localhost origin and callback remain configured for local testing. Do not
remove them unless local Google sign-in is intentionally retired.

The scheme, host, port, path, and trailing-slash behavior must match exactly.
Remove localhost from the production OAuth client or use a separate local
client. Confirm the OAuth consent screen, domain verification, administrator
account/test-user status, and any Google-required homepage, privacy-policy, and
terms links before publishing the OAuth application. Legal-policy text must be
provided and approved by the site owner; it is not generated by this project.

## Database release sequence

Do not run `prisma migrate dev`, `db push`, `reset`, or the seed as part of a
serverless request or automatic application startup.

1. Back up or create a Neon restore point according to the account's plan.
2. Confirm `DATABASE_URL` is pooled and `DIRECT_URL` is direct, with both
   targeting the same approved branch, role, database, and schema.
3. Run `pnpm db:validate` and `pnpm db:generate` locally or in CI.
4. Run `pnpm db:migrate:status` against the exact production target.
5. If a checked-in migration is pending, run `pnpm db:migrate:deploy` once from
   a controlled release job using `DIRECT_URL`.
6. Run the explicit idempotent seed only for a new approved database. Do not
   run it automatically on every deployment.
7. Recheck migration status before directing traffic to the deployment.

Runtime Prisma and Better Auth traffic uses only the pooled `DATABASE_URL`.
Where the platform supports separate scopes, expose `DIRECT_URL` only to the
controlled release/migration job rather than to request-serving functions.

The active Neon database has both checked-in migrations applied:

- `20260808210000_phase5_initial`
- `20260814180000_phase6_authentication`

The local and database migration inventories match. Phase 10 did not apply a
migration, run the seed, create a database, or replace any portfolio records.
Production requests use the pooled connection; direct access remains confined
to controlled migration diagnostics and releases.

The currently pinned `pg` 8.x runtime warns that its historical handling of
`sslmode=require` will change in the next major release. Keep Neon's generated
TLS parameters for this deployment; before any future `pg` 9 upgrade, validate
and explicitly select the intended certificate-verification semantics for both
database URLs.

## Deploy, redeploy, and roll back

Use GitHub as the authoritative production release path:

1. Create or update a topic branch; do not work directly on `main`.
2. Run the pre-deployment checks and push the reviewed commit.
3. Open a pull request into `main`, review the complete diff, and confirm it
   contains no secrets or generated artifacts.
4. Wait for all required GitHub checks and the Vercel Preview deployment to
   pass.
5. Merge the pull request only after approval and green checks.
6. Confirm Vercel deploys the exact merged `main` commit to the production
   domain, then complete the post-deployment smoke tests.

Reserve a manual Vercel CLI production deployment or promotion for an explicitly
approved exceptional release or recovery. Inspect the immutable deployment
before promoting it:

```powershell
pnpm dlx vercel@latest deploy --prod --skip-domain --yes
pnpm dlx vercel@latest inspect https://DEPLOYMENT_URL --wait --logs
pnpm dlx vercel@latest promote https://DEPLOYMENT_URL --yes
```

Do not put environment values on the command line. They are read from the
encrypted Vercel Production environment. A normal redeploy follows the same
topic-branch and pull-request workflow so the merged `main` commit remains the
audited source of the production release.

To roll back, open the Vercel project's **Deployments** page, select the prior
known-good immutable deployment, inspect its status, and promote it to the
production domain. The equivalent CLI operation is:

```powershell
pnpm dlx vercel@latest promote https://PREVIOUS_READY_DEPLOYMENT_URL --yes
```

Rollback changes the code deployment, not database state. If a future release
includes an incompatible migration, prepare and approve a separate database
recovery plan before deploying it. Never use `migrate reset`, `db push`, or an
automatic seed as a rollback mechanism.

## Pre-deploy checks

Run from a clean checkout with the intended production environment names
available without printing their values:

```bash
pnpm install --frozen-lockfile
pnpm db:validate
pnpm db:generate
pnpm test
pnpm lint
pnpm typecheck
pnpm build
pnpm start
```

Verify `/robots.txt`, `/sitemap.xml`, canonical metadata, and the security
headers. Confirm `/admin`, `/api/auth`, `/sign-in`, and `/access-denied` return
no-store/no-index headers.

## Post-deploy smoke test

1. Open Home, About, Research and its eligible detail, Projects and its
   eligible detail, Experience, Writing, Resume, and Contact.
2. Confirm an unknown research/project slug returns 404.
3. Inspect title, description, canonical, robots, and sitemap output without
   copying secrets or OAuth callback URLs.
4. Confirm signed-out `/admin` redirects to `/sign-in`.
5. Sign in with the exact `ADMIN_EMAIL`; verify another identity is denied.
6. Make a harmless reversible admin edit, publish it, confirm public cache
   invalidation, then restore the exact original record.
7. Confirm a draft/unpublished record and an archived record remain absent
   from HTML, metadata, sitemap, and public detail lookup.
8. Sign out and confirm the database session is revoked before the browser is
   returned to `/sign-in`.
9. Temporarily test an unavailable database only in a controlled environment;
   confirm the last-known-good/static fallback renders without internal error
   details.

## Phase 10 production verification record

The first Vercel production deployment was created on August 29, 2026, and the
final owner-operated authentication/content check was completed before this
guide was finalized. The following results were verified:

- Vercel installed the pinned pnpm release through Corepack, generated Prisma
  Client, compiled Next.js, completed TypeScript checking, and marked the
  deployment **Ready** before the stable domain was promoted.
- Home, About, Research, Projects, Experience, Writing, Resume, Contact,
  Sign-in, robots, sitemap, and the eligible research/project detail routes
  return successfully over HTTPS.
- Unknown research and project slugs return 404.
- Neon pooled reads succeed. The production sitemap contains the eligible
  database-backed research and project slugs, and migration inventory matches
  the two checked-in migrations.
- Google sign-in completes for the exact `ADMIN_EMAIL`, establishes the Better
  Auth session, and opens the protected dashboard.
- Signed-out `/admin` redirects server-side to `/sign-in` with `no-store` and
  `noindex` protections. The owner confirmed sign-out removed dashboard access
  and returned the flow to **Continue with Google**.
- The owner changed the published LinkedIn label from `LinkedIn` to
  `LinkedIn Test`, saved it, verified persistence in the production admin
  record, and restored the exact original label. A final database and public
  read confirms `LinkedIn`; `LinkedIn Test` is absent.
- The current database has no draft, archived, or future-dated research/project
  fixtures. Their exclusion is therefore covered by repository predicates and
  automated privacy regression tests rather than by creating production test
  content.
- HTTP redirects to HTTPS. The production response includes HSTS, frame denial,
  MIME-sniffing protection, a strict-origin referrer policy, and a restrictive
  permissions policy. Admin/auth routes remain excluded by robots metadata.
- Targeted public-response and source-control checks found no exposed
  credentials, connection strings, private environment files, Prisma errors,
  or internal error details.

A strict Content Security Policy, Cross-Origin-Opener-Policy, and
Cross-Origin-Resource-Policy are not currently emitted. This is a documented,
non-blocking hardening opportunity; a CSP should be introduced only with
deployment-specific validation of Next.js bootstrap scripts and Google OAuth.

## Operational boundaries

- The process-local last-known-good store is opportunistic and not shared
  between serverless instances. Next's tagged cache and the reviewed static
  fallback remain the production safety net for this phase.
- Better Auth's memory rate limiter is per runtime instance. Keep the exact
  administrator allowlist and server-side guards; add provider-level abuse
  controls before higher-traffic or multi-user use.
- The Vercel-generated production domain currently serves HSTS after successful
  HTTPS verification. Re-evaluate HSTS and `includeSubDomains` before adding a
  custom domain or new subdomains; do not preload an unverified custom domain.
- A strict nonce-based Content Security Policy is intentionally not introduced
  in this phase; it requires deployment-specific validation of Next.js inline
  bootstrap scripts and OAuth behavior.
