# Feature-module boundaries

`src/features` contains the domain-oriented public presentation introduced in
Phase 3, the detail records added in Phase 4, the authorized admin feature from
Phase 6, the server-only public repositories introduced in Phase 7, and the
end-to-end administrator workflows completed in Phase 8. The verified static
adapters remain as the final operational-error fallback.

## Dependency rules

- `src/app` owns routing, layouts, metadata, and route-level composition. Keep domain rules out of route files.
- A feature owns its domain types, validation, presentation, and—when Phase 7 requires it—server query and repository boundaries.
- Features may import domain-neutral UI from `src/components/ui` and shared infrastructure from `src/lib`.
- Features must not import from `src/app` or reach into another feature's private files.
- Components and browser code must never import Prisma, database credentials, or generated database types.
- Server repositories may call the lazy client in `src/lib/db.ts`; they must map rows into feature-owned types before returning data.
- Public queries must filter `PUBLISHED` records and apply deterministic ordering. Draft, archived, and preview reads require future server-side authorization.
- Provider details, connection strings, and Prisma driver types stay inside the infrastructure boundary.

A feature can grow toward this shape when its dynamic integration phase begins:

```text
feature-name/
├── components/       Feature-specific presentation and interaction
├── schemas.ts        Input and content validation
├── types.ts          Database-independent domain and view types
├── queries.server.ts Server-only public read/cache orchestration
├── actions.ts        Authorized and validated mutations (Phase 6+)
└── repository.server.ts Prisma-backed public persistence adapter
```

Do not add all files to every feature preemptively. Phase 7 adds repositories
only where a public route consumes that domain.

## Phase 5 database boundary

| Path | Responsibility |
| --- | --- |
| `prisma/schema.prisma` | Persistence models, enums, relationships, indexes, and PostgreSQL mappings. |
| `prisma/migrations/` | Versioned SQL, including constraints Prisma cannot express directly. |
| `prisma/seed-data.ts` | Reviewed CV-backed and user-approved bootstrap records only. |
| `prisma/seed.ts` | Explicit, idempotent transaction that upserts the bootstrap records. |
| `prisma.config.ts` | Prisma 7 schema, migration, direct connection, optional shadow connection, and seed configuration. |
| `src/generated/prisma/` | Generated client code; infrastructure only and never edited manually. |
| `src/lib/env.server.ts` | Lazy validation for the private pooled runtime URL. |
| `src/lib/db.ts` | Lazy server-only `PrismaClient` singleton using the PostgreSQL adapter. |

The runtime uses pooled `DATABASE_URL`; Prisma CLI migration and seed commands
use direct `DIRECT_URL`. An optional `SHADOW_DATABASE_URL` supports migration
development. Local credentials live in ignored `.env.local`; set
`DOTENV_CONFIG_PATH=.env.local` before connected Prisma CLI commands.

The owner-approved Neon database contains the Phase 5 content migration, the
additive Phase 6 auth migration, and the explicit reviewed bootstrap records.
Static builds, schema validation, and client generation do not need database
access; runtime public reads, admin sessions, and CRUD operations do.

## Domain status

| Feature | Phase 7 public source |
| --- | --- |
| `home` | Published profile, links, experience, education, research, selected project, and skills. |
| `profile` | Published Profile/About, experience, education, research, skills, eligible resume metadata, and published certifications. |
| `research` | Eligible research directions/questions/interests and eligible publications; unsupported evidence remains absent. |
| `projects` | Eligible projects with ordered contributions, features, technologies, and safe repository links. |
| `experience` | Published client engagements, education, skill categories, and skills. |
| `writing` | The eligible current research direction only; Article detail/rendering remains outside the current phase. |
| `contact` | Published profile location/social links and an eligible selected-project repository; ContactSubmission remains deferred beyond Phase 9. |
| `media` *(persistence only)* | Provider-neutral metadata and typed placements exist in the schema; no provider, asset, upload, or absolute local path is configured. |

The schema also supports certifications, formal publications, resume assets,
site settings, research/project relationships, experience/skill relationships,
and content media. Those records or relationships stay empty where the source
material does not establish them.

## Seed contract

The explicit `pnpm db:seed` command upserts only reviewed content:

- 1 profile and 2 social/contact links
- 4 research interests
- 1 early-stage research direction and 5 ordered questions
- 1 academic capstone, 4 ordered contribution groups, 5 ordered features, and 6 technologies
- 2 client engagements and 2 education records with month/year precision
- 7 skill categories and 40 skills

Seeded domain records use immutable nullable `seedKey` values that are separate
from public slugs. Future admin-created records may leave `seedKey` null, while
an approved public slug can change without changing seed identity.

CV-backed public records carry `ContentSource.ACADEMIC_CV`; the approved
visitor-facing profile narrative carries `ContentSource.USER_PROVIDED`. Both
use `ContentStatus.PUBLISHED`. A future record marked `ADMIN` is preserved by
seed reruns. These fields describe provenance and portfolio visibility; they do
not convert the early-stage research direction into a publication or claim
that unsupported project results exist.

The two seeded records with a scheduling field—the research direction and
capstone—receive one bootstrap `publishedAt` timestamp. Reruns preserve that
timestamp. Phase 7 treats a null or future publication timestamp as private.

The following remain unseeded: publications, certifications, articles, tags,
media assets and placements, resume records/files, site settings, contact
submissions, project results or dates, research findings or dates, GPA,
coursework, thesis details, and unsupported cross-domain relationships.

## Phase 6/8 admin boundary

`src/features/admin` owns a closed resource registry, database-independent DTOs,
strict Zod schemas, server-only Prisma repositories and queries, and protected
Server Actions. The private UI lives under `src/app/admin` and
`src/components/admin`.

Every exported admin query and action revalidates the database-backed session
with `assertAdmin()` before parsing input or accessing data. Routes also use
`requireAdmin()`, while `src/proxy.ts` supplies an additional early protection
layer. Input schemas never accept `profileId`, `source`, `seedKey`, timestamps,
auth fields, or arbitrary model names. Ownership and relations are derived and
checked on the server.

The client sign-out control uses Better Auth to revoke the single-owner
administrator's database sessions before clearing its signed cookie. A failed
database revocation is surfaced to the administrator and leaves the UI signed
in instead of silently abandoning an active server-side session.

The admin feature supports profile creation/update and reusable
create/read/update/publish/unpublish/archive workflows for the allowlisted
portfolio resources. Archival is reversible and promotes records to `ADMIN`
ownership, preventing an explicit seed rerun from restoring or overwriting a
managed record. Site settings remain closed until specific keys are allowlisted.
Public pages do not import this admin data layer.

Phase 8 adds parent-scoped transactional editing for `ResearchQuestion` and the
supported `ProjectContribution`, `ProjectFeature`, and `ProjectTechnology`
children. Ordered arrays are validated as complete aggregates, the parent
`updatedAt` value supplies optimistic concurrency, and removal of a persisted
child requires an exact server-derived removal set plus explicit confirmation.
Nested child keys/slugs remain structural identifiers rather than visitor-facing
claims. No schema or migration change is required.

Publishing and archival are separate actions. Generic create/edit and quick
visibility actions accept only `DRAFT` or `PUBLISHED`; only the confirmed archive
action can set `ARCHIVED`. For models with `publishedAt`, the server assigns a
timestamp on first publish, retains it while published, and clears it on draft
or archive. Every successful mutation clears affected last-known-good snapshots,
updates the corresponding public cache tags, and revalidates the relevant admin
and public paths.

## Phase 7 public repository contract

Phase 7 repositories:

- run only on the server;
- select only the fields required by their feature;
- filter public content by `PUBLISHED` status and require a non-null,
  non-future publication time where applicable;
- order records using stored `sortOrder` plus a stable tie-breaker;
- translate partial year/month fields without fabricating a calendar day;
- return feature-owned domain types rather than Prisma payload types;
- use `notFound()` for missing or non-public detail records;
- preserve the current honest empty states when a table has no verified rows.

Repository reads are cached for five minutes with domain tags. Successful admin
mutations invalidate affected tags immediately. A successful empty query is
authoritative; fallback content is used only after an operational read failure,
first from a bounded process-local last-known-good snapshot when present, then
from the existing reviewed static adapter. Public routes never import the
authorized admin repository.

All factual records continue to follow the root README's CV-based integrity
rules. A nullable column or empty table is intentional architecture, not an
invitation to synthesize content.
