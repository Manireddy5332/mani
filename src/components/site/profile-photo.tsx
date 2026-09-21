"use client";

import Image from "next/image";
import { useState } from "react";

import type { PublicProfilePhoto } from "@/features/profile-photo/types";
import { cn } from "@/lib/cn";

type PhotoVariant = "card" | "hero";

function PhotoImage({
  photo,
  variant,
}: {
  readonly photo: PublicProfilePhoto;
  readonly variant: PhotoVariant;
}) {
  const [unavailable, setUnavailable] = useState(false);

  if (unavailable) return null;

  return (
    <div
      className={cn(
        "relative mx-auto max-w-full rounded-[1.75rem] border border-primary/20 bg-canvas/70 p-2 shadow-sm",
        variant === "hero"
          ? "mb-10 w-56 sm:w-64 lg:w-72"
          : "my-7 w-40 sm:w-48",
      )}
    >
      <Image
        alt={photo.alt}
        className="aspect-square h-auto w-full rounded-3xl object-cover"
        height={photo.height}
        onError={() => setUnavailable(true)}
        src={photo.src}
        width={photo.width}
        // Uploads are already optimized. Bypass the persistent image optimizer
        // so removed or unpublished photos still pass the live delivery gate.
        unoptimized
      />
    </div>
  );
}

export function ProfilePhoto({
  photo,
  variant = "card",
}: {
  readonly photo: PublicProfilePhoto | null;
  readonly variant?: PhotoVariant;
}) {
  if (!photo) return null;

  return <PhotoImage key={photo.src} photo={photo} variant={variant} />;
}
