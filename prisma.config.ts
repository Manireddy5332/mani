import "dotenv/config";

import { defineConfig } from "prisma/config";

const shadowDatabaseUrl = process.env.SHADOW_DATABASE_URL;

export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: {
    path: "prisma/migrations",
    seed: "tsx prisma/seed.ts",
  },
  datasource: {
    // Prisma CLI commands use a direct PostgreSQL connection. Commands that do
    // not connect (validate/generate) remain usable before credentials exist.
    url: process.env.DIRECT_URL!,
    ...(shadowDatabaseUrl ? { shadowDatabaseUrl } : {}),
  },
});
