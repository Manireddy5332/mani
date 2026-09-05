import { toNextJsHandler } from "better-auth/next-js";

import { getAuth } from "@/lib/auth";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function handlers() {
  return toNextJsHandler(getAuth());
}

export function GET(request: Request): Promise<Response> {
  return handlers().GET(request);
}

export function POST(request: Request): Promise<Response> {
  return handlers().POST(request);
}
