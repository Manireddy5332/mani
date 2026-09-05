# Manikanta Reddy Anugu — Academic + Professional Portfolio

A production-oriented academic and professional portfolio for presenting verified work across AI/ML engineering, Generative AI, and research. The application uses the Next.js App Router with a server-only, database-backed public content layer and a private content-management system.

## Project status: Phase 10 (deployed)

Phases 2–8 established the design system, PostgreSQL persistence, private
administrator area, database-backed public portfolio, and supported end-to-end
content-management workflow. Phase 9 added the production-readiness boundary,
and Phase 10 deployed and verified that architecture without redesigning or
replacing it.

Production is hosted on Vercel at
[https://manikanta-ai-portfolio-pi.vercel.app](https://manikanta-ai-portfolio-pi.vercel.app).
The deployment uses the existing Neon database and Google OAuth configuration;
no new database, seed run, or destructive migration was performed for release.

The completed application includes:

- Next.js 16, React 19, strict TypeScript, Tailwind CSS 4, and ESLint
- responsive light and dark themes with reduced-motion support
- dedicated About, Research, Projects, Experience, Writing, Resume, and Contact routes
- runtime research and project detail routes with strict public-slug handling
- Prisma ORM 7.9.1 with the PostgreSQL driver adapter
- a normalized portfolio schema with content status, source provenance, ordering, relationships, and media references
- month/year date fields and PostgreSQL checks that retain the precision of the Academic CV
- a checked-in initial migration and an idempotent, CV-backed seed command
- a lazy, server-only Prisma client and private environment validation
- Better Auth with Google OAuth, database-backed sessions, encrypted OAuth
  access/refresh tokens, and stored-ID-token suppression
- exact, verified `ADMIN_EMAIL` authorization at the identity, session, route,
  query, and mutation boundaries
- a protected `/admin` dashboard with reusable forms, lists, explicit
  publish/unpublish controls, featured flags, ordering, and confirmed archival
- ordered, transactional editors for research questions and project
  contributions, features, and technologies
- strict Zod input validation, server-owned relationships, safe Prisma writes,
  and content-source protection for later seed runs
- feature-owned, server-only public repositories that return serialized DTOs
- strict `PUBLISHED` filtering, with a non-null, non-future `publishedAt`
  required for schedulable public records
- deterministic ordering, tagged five-minute caching, and immediate public
  cache/path invalidation after successful administrator mutations
- last-known-good content when available and the original verified static
  adapters as an operational-error fallback only
- dynamic database metadata and detail lookup for published research and projects
- exact HTTPS-or-loopback application-origin validation for public metadata and
  Better Auth, with matching public/auth origins
- explicit robots and database-backed sitemap metadata that exclude every
  private surface
- conservative security headers and no-store/no-index headers on private/auth
  routes
- a verified Vercel production deployment using Node.js 24 and the pinned pnpm
  release
- production Google OAuth, exact-email administrator authorization, Neon
  connectivity, security headers, and public/private route checks
- a production deployment and rollback guide covering Vercel, Google OAuth,
  Neon, Prisma migrations, build validation, and post-deploy privacy checks

The configured Neon database contains the Phase 5 portfolio schema and the
Phase 6 Better Auth tables. The explicit idempotent seed has populated the
reviewed CV-backed portfolio records. Public body routes now read eligible
content through the Phase 7 repositories; the static adapters remain intact as
the final fallback for an operational database/cache failure.

## Prerequisites

- [Node.js](https://nodejs.org/) 24 or newer
- [pnpm](https://pnpm.io/) 11
- PostgreSQL for authentication, admin reads/writes, migrations, seeding, or Prisma Studio
- a Google Cloud OAuth Web application configured for the local and production
  administrator sign-in origins

Confirm the installed runtime:

```bash
node --version
pnpm --version
```

## Run the portfolio locally

The public site can be viewed without signing in:

```powershell
cd D:\manikanta-ai-portfolio
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The Prisma client is
created only when server-side database, public content, or authentication code
calls it. If a public read is temporarily unavailable, the server uses cached
last-known-good data when possible and then the reviewed static fallback. Open
[http://localhost:3000/sign-in](http://localhost:3000/sign-in) for the private
Google sign-in flow and [http://localhost:3000/admin](http://localhost:3000/admin)
for the protected dashboard.

To test the optimized application locally:

```powershell
pnpm build
pnpm start
```

## Configure the private environment

Create an ignored project-root `.env.local` file from the safe template:

```powershell
Copy-Item .env.example .env.local
```

Replace placeholders only with credentials from services you control. Never
commit `.env.local`, paste its values into source files, or expose private
values through a `NEXT_PUBLIC_` variable.

Next.js loads `.env.local` for the application runtime. For Prisma CLI commands,
set `DOTENV_CONFIG_PATH=.env.local` in the current PowerShell process first:

```powershell
$env:DOTENV_CONFIG_PATH = ".env.local"
```

### Neon

For a manually created Neon database, use the two connection strings supplied
by the Neon console:

- `DATABASE_URL` is the pooled application URL. Its hostname normally includes `-pooler`.
- `DIRECT_URL` is the non-pooled URL used by Prisma migrations and explicit seeding.
- `SHADOW_DATABASE_URL` is optional. When using `pnpm db:migrate:dev`, it can point to a separate empty database or Neon branch used only as Prisma's shadow database.

Keep Neon-provided TLS query parameters intact. Prisma migrations use only the
direct URL; runtime application and Better Auth queries use only the pooled URL.

### Local PostgreSQL

For a local PostgreSQL server, `DATABASE_URL` and `DIRECT_URL` may use the same
direct connection string. Create an empty application database first. If
`migrate dev` requires a shadow database, create a second empty database and set
`SHADOW_DATABASE_URL` to it. Do not point the shadow URL at a database that
contains data you need.

Example shape only:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/portfolio
DIRECT_URL=postgresql://USER:PASSWORD@localhost:5432/portfolio
# SHADOW_DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/portfolio_shadow
```

### Better Auth and Google OAuth

The private administrator flow expects these server-only names:

- `BETTER_AUTH_SECRET`: a high-entropy value of at least 32 characters
- `BETTER_AUTH_URL`: `http://localhost:3000` for local development
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`: a Google OAuth Web client
- `ADMIN_EMAIL`: the one verified Google address allowed into `/admin`

In the Google Cloud OAuth client, configure:

- authorized JavaScript origin: `http://localhost:3000`
- authorized redirect URI: `http://localhost:3000/api/auth/callback/google`

Better Auth is self-hosted and needs no Better Auth account. A Google account
that authenticates successfully but does not exactly match the normalized,
verified `ADMIN_EMAIL` is denied server-side.

To test the private area locally:

1. Run `pnpm dev` and open `http://localhost:3000/sign-in`.
2. Choose **Continue with Google** and use the exact verified account configured
   by `ADMIN_EMAIL`.
3. Confirm the callback returns to `http://localhost:3000/admin`.
4. Review the dashboard and use **Sign out** to revoke the administrator's
   database sessions before the local cookie is cleared.

An unauthenticated request to `/admin` redirects to `/sign-in`. A verified
Google identity that does not match the allowlist cannot create an admin
session. Do not add a password bypass or development-only administrator route.

## Database commands

| Command | Purpose | Connects to PostgreSQL? |
| --- | --- | --- |
| `pnpm db:validate` | Validate `prisma/schema.prisma`. | No |
| `pnpm db:generate` | Regenerate the typed client in `src/generated/prisma`. | No |
| `pnpm db:format` | Format the Prisma schema. | No |
| `pnpm db:migrate:dev` | Create/apply a development migration using the direct URL. | Yes |
| `pnpm db:migrate:deploy` | Apply checked-in migrations in a controlled environment. | Yes |
| `pnpm db:migrate:status` | Compare checked-in migrations with the target database. | Yes |
| `pnpm db:seed` | Explicitly run the idempotent Academic-CV seed. | Yes |
| `pnpm db:studio` | Open Prisma Studio for the configured database. | Yes |

`pnpm install` also runs `prisma generate` through `postinstall`. Prisma 7 does
not run the seed automatically during migration or reset; seeding happens only
when `pnpm db:seed` is invoked intentionally.

A safe first database setup is:

```powershell
pnpm db:validate
pnpm db:generate
pnpm db:migrate:deploy
pnpm db:seed
```

Use `db:migrate:dev` only when authoring a new migration against a disposable
development database. Use `db:migrate:deploy` to apply the checked-in migration
without generating schema changes.

## Quality commands

```bash
pnpm db:validate
pnpm db:generate
pnpm test
pnpm lint
pnpm typecheck
pnpm build
```

These checks do not require a database connection. Migration status, applying
migrations, seeding, and Studio do require valid credentials.

## Production deployment

The application is deployed to Vercel at
[https://manikanta-ai-portfolio-pi.vercel.app](https://manikanta-ai-portfolio-pi.vercel.app).
The production environment contract, Google OAuth configuration, controlled
Prisma release sequence, Vercel settings, verification evidence, and
redeploy/rollback procedures are documented in
[`docs/production-deployment.md`](docs/production-deployment.md).

Vercel is linked to `Manireddy5332/mani`, but this local workspace currently
has no commit or Git remote. The Phase 10 release was uploaded directly from
the reviewed workspace with Vercel CLI. Review, commit, and push this exact tree
before treating Git-triggered deployments as authoritative.

Production must use one canonical HTTPS origin for both
`NEXT_PUBLIC_SITE_URL` and `BETTER_AUTH_URL`. Runtime database/auth traffic uses
the pooled Neon `DATABASE_URL`. Vercel request-serving functions do not receive
`DIRECT_URL`; controlled Prisma migration commands use the matching direct URL
outside the runtime environment. Localhost values, local OAuth secrets, and
local database credentials must not be copied into production.

## Schema, migration, and seed strategy

`prisma/schema.prisma` models profile content, social links, research interests
and projects, publications, projects and their structured case-study records,
experience, education, skill categories and skills, certifications, articles,
tags, media, resume versions, contact submissions, and site settings.

The initial content migration is checked into
`prisma/migrations/20260808210000_phase5_initial`. The additive authentication
migration is checked into
`prisma/migrations/20260814180000_phase6_authentication`; it creates Better
Auth's `user`, `session`, `account`, and `verification` tables, eight supporting
indexes, and two cascading account/session ownership foreign keys. It does not
alter or remove Phase 5 content data.

In addition to Prisma's
tables, relations, indexes, and foreign keys, it contains PostgreSQL checks for:

- valid month values and chronological month/year ranges
- current records having no end date
- research progress between 0 and 100
- valid GPA relationships without supplying a GPA
- positive article reading time and media dimensions
- one current resume per profile

CV month/year dates are stored as separate integer fields. The migration does
not fabricate a day or convert a partial date into an inaccurate timestamp.

`prisma/seed.ts` uses immutable internal `seedKey` values, separate public
slugs, and `upsert` inside a transaction. A public slug can therefore change
without changing which record an explicit seed run reconciles. CV-backed
domain records use the `ACADEMIC_CV` source; the
visitor-facing profile narrative approved during Phase 2 uses `USER_PROVIDED`.
Records later marked `ADMIN` are not overwritten by a seed rerun. The seed
contains exactly:

- one profile and two contact/social links
- four research interests
- one early-stage research direction and five questions
- one academic capstone with four contribution groups, five features, and six technologies
- two client engagements and two education records
- seven skill categories and forty skills

Publications, certifications, articles, tags, media assets, resume files, site
settings, and contact submissions are intentionally left empty. The seed does
not copy the Academic CV into the project, expose the phone number, create a
GitHub-profile claim, invent results or publications, or store an absolute
Windows path. Run the seed only for intentional bootstrap or development setup.
It reconciles the checked-in source records but does not delete unrelated data
or replace records that a future admin workflow has taken ownership of.

## Authentication and authorization

Better Auth is initialized lazily on the server and exposes its standard route
handler at `/api/auth/[...all]`. The browser begins a Google OAuth flow through
the same-origin Better Auth client; credentials never enter a Client Component.
Sessions live in PostgreSQL. Sign-out first calls Better Auth's error-reporting
session-revocation endpoint and only clears the browser cookie after revocation
succeeds, preventing a database failure from being reported as a successful
sign-out.

Authorization is deliberately stricter than authentication:

- the Google identity must be verified and exactly match normalized `ADMIN_EMAIL`;
- account- and session-creation hooks reject every other identity;
- Google identity information is refreshed on each sign-in;
- `/admin` is checked by the Next.js proxy and again by its server layout/pages;
- every admin query and Server Action independently revalidates the live session;
- protected inputs cannot supply ownership IDs, source provenance, seed keys,
  timestamps, or auth fields.

Access and refresh tokens are encrypted at rest. Stored Google ID tokens are
explicitly discarded because the portfolio does not use them. Account linking
is disabled. Better Auth's in-process rate limiter is enabled, but its state is
local to each Vercel runtime instance. Distributed rate-limit storage must be
selected before deliberate horizontal scaling or materially higher traffic.

## Admin content management

The protected dashboard uses one closed, typed registry and explicit
Prisma-backed handlers for Profile/About, social links, research interests,
research projects, projects, experience, education, publications, writing,
resume metadata, skill categories, skills, and certifications. Site settings
remain read-only until individual keys receive dedicated schemas.

Create and edit forms support strict validation, draft/published visibility,
featured flags, numeric ordering, relevant relationships, pending states,
field errors, and success feedback. Publishing is explicit; schedulable records
receive a server-owned `publishedAt` timestamp, while unpublishing or archiving
clears it. Editing promotes a record to `ADMIN` ownership so an intentional
seed rerun cannot overwrite it.

Research questions and the supported project contribution, feature, and
technology children are managed as ordered parent-owned aggregates. Their saves
are transactional, revision-checked, and require confirmation before a saved
child is permanently removed. Top-level destructive actions remain reversible
archives; physical deletion is intentionally unavailable for CV-backed records,
and Profile deletion is blocked. Successful writes invalidate the precise
public cache tags and paths affected by the resource, so an approved published
change is visible without a source edit. File uploads, media providers, rich
article/detail rendering, arbitrary site-setting writes, and contact ingestion
remain outside this phase.

## Architecture and folders

```text
manikanta-ai-portfolio/
├── prisma/
│   ├── migrations/             Versioned PostgreSQL migration SQL
│   ├── schema.prisma           Database models, enums, indexes, and relations
│   ├── seed-data.ts            Reviewed CV-backed and user-approved records
│   └── seed.ts                 Explicit idempotent seed command
├── public/                      Static files served directly
├── src/
│   ├── app/                     App Router layouts, pages, and metadata
│   │   ├── admin/               Protected dashboard routes
│   │   └── api/auth/            Better Auth request handler
│   ├── components/              Shared public, admin, and design-system components
│   ├── features/                Domain presentation, public repositories, and admin boundaries
│   ├── generated/prisma/        Generated Prisma client; do not edit manually
│   └── lib/
│       ├── auth/                Server auth, policy, and authorization guards
│       ├── db.ts                Lazy server-only Prisma singleton
│       ├── env.server.ts        Private runtime environment validation
│       └── public-content/      Cache tags, fallback policy, dates, and eligibility helpers
├── .env.example                Safe connection-string templates
├── prisma.config.ts            Prisma 7 CLI, migration, and seed configuration
└── package.json                Application and database scripts
```

Public routes and browser components do not import Prisma directly. Phase 6
admin repositories remain authorized and private. Phase 7 public repositories
call `getDatabase()` only on the server, select minimal fields, map rows into
feature-owned domain types, and enforce public visibility. Generated Prisma
types remain infrastructure details.

See [`src/features/README.md`](src/features/README.md) for the active feature
boundaries and the Phase 7 public/Phase 8 administrator contracts.

## Content integrity

The supplied Academic CV remains the source of truth for personal, academic,
and professional facts:

- Do not invent or infer employment, research, publications, projects, certifications, awards, metrics, or achievements.
- Preserve exact role, organization, education, and month/year information unless Manikanta provides a newer verified source.
- Label work accurately by stage and keep draft or archived records out of public queries.
- Omit unknown values instead of filling nullable database fields with plausible content.
- Publish professional metrics only after a separate confidentiality review.

The existence of a schema table is not evidence that a corresponding record
exists. Empty publication, certification, writing, resume, media, and contact
tables must remain empty until verified content or real submissions are added.

## Design-system principles

The visual direction is an original premium editorial identity for a personal academic and professional portfolio. “Evidence Atlas” remains a focused research-mapping component within that broader system—not a reproduction of the previous portfolio and not the identity of the entire homepage.

- Semantic tokens express intent (`surface`, `foreground`, `accent`, `border`) instead of page-specific colors.
- Reusable, typed primitives own common states and variants; pages compose them rather than duplicating markup and styles.
- The warm-paper/light and blue-black/dark themes share one hierarchy and maintain readable contrast.
- Typography separates editorial emphasis, interface copy, and technical metadata without relying on decorative effects.
- Layouts begin with semantic HTML and responsive CSS; enhancement must preserve keyboard and screen-reader access.
- Motion is restrained, transform/opacity based, and disabled or simplified for `prefers-reduced-motion`.
- The foundation avoids heavy visualization libraries, scroll-jacking, fabricated counters, and technology-logo walls.

## Phase boundaries

- **Phase 5 (complete):** Prisma/PostgreSQL schema, checked-in migration, CV-backed bootstrap seed, and lazy server-only database infrastructure.
- **Phase 6 (complete):** Google authentication, exact-email authorization, protected admin dashboard, and reusable validated CRUD foundation.
- **Phase 7 (complete):** server-only public repositories, dynamic public-content integration, visibility filtering, caching, and verified fallback handling.
- **Phase 8 (complete):** end-to-end administrator publishing, nested research/project content management, public cache revalidation, and lifecycle/privacy regression coverage.
- **Phase 9 (complete):** production environment validation, deployment configuration guidance, security hardening, SEO discovery routes, and production build/runtime verification.
- **Phase 10 (complete):** Vercel deployment, production Google OAuth and Neon verification, public/private route smoke testing, reversible admin-to-public verification, and release documentation.
- **Later phases:** contact ingestion, file/media storage, rich article/detail rendering, advanced motion, structured-data expansion, and observability.

No contact-form storage, file uploads, rich article/detail engine, hosted
analytics, custom domain purchase, or deferred feature work was performed in
Phase 10.
