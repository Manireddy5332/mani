const LOOPBACK_HOSTNAMES = new Set(["localhost", "127.0.0.1", "[::1]"]);

/**
 * Public and authentication base URLs must be exact origins. Production hosts
 * use HTTPS; HTTP remains available only for local loopback development.
 */
export function isSecureApplicationOrigin(value: string): boolean {
  if (value !== value.trim()) return false;

  try {
    const url = new URL(value);
    const isHttp = url.protocol === "http:";
    const isHttps = url.protocol === "https:";
    const isOriginOnly =
      url.username === "" &&
      url.password === "" &&
      url.pathname === "/" &&
      url.search === "" &&
      url.hash === "";

    return (
      (isHttps || (isHttp && LOOPBACK_HOSTNAMES.has(url.hostname))) &&
      isOriginOnly
    );
  } catch {
    return false;
  }
}

export function applicationOriginsMatch(
  first: string,
  second: string,
): boolean {
  try {
    return new URL(first).origin === new URL(second).origin;
  } catch {
    return false;
  }
}
