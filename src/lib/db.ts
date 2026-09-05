import "server-only";

import { PrismaPg } from "@prisma/adapter-pg";

import { PrismaClient } from "@/generated/prisma/client";
import { getServerEnvironment } from "@/lib/env.server";

const globalForDatabase = globalThis as typeof globalThis & {
  portfolioDatabase?: PrismaClient;
};

function createDatabaseClient() {
  const { DATABASE_URL } = getServerEnvironment();
  const adapter = new PrismaPg({
    connectionString: DATABASE_URL,
    connectionTimeoutMillis: 10_000,
    idleTimeoutMillis: 10_000,
    max: process.env.NODE_ENV === "production" ? 5 : 3,
  });

  return new PrismaClient({
    adapter,
    log: process.env.NODE_ENV === "development" ? ["warn", "error"] : ["error"],
  });
}

/**
 * Returns the server-only Prisma singleton. The client is intentionally lazy so
 * schema generation and static builds do not require database credentials.
 */
export function getDatabase(): PrismaClient {
  globalForDatabase.portfolioDatabase ??= createDatabaseClient();
  return globalForDatabase.portfolioDatabase;
}

export type DatabaseClient = PrismaClient;
