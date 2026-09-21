import type { NextConfig } from "next";

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value:
      "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()",
  },
] as const;

const privateRouteHeaders = [
  { key: "Cache-Control", value: "no-store, max-age=0" },
  { key: "X-Robots-Tag", value: "noindex, nofollow, noarchive" },
] as const;

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  images: {
    // CMS portraits are upload-optimized and publication-gated on every request.
    // Do not let the shared image optimizer retain a removed or unpublished photo.
    localPatterns: [{ pathname: "/_next/static/media/**", search: "" }],
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [...securityHeaders],
      },
      ...["/admin/:path*", "/api/auth/:path*", "/sign-in", "/access-denied"].map(
        (source) => ({
          source,
          headers: [...privateRouteHeaders],
        }),
      ),
    ];
  },
};

export default nextConfig;
