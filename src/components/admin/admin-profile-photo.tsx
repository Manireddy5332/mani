"use client";

import { Camera, Save, Trash2 } from "lucide-react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

import { Button } from "@/components/ui";
import type { AdminProfilePhotoState } from "@/features/profile-photo/types";
import { cn } from "@/lib/cn";

type Feedback = { message: string; error: boolean };
type SelectedPhoto = { file: File; preview: string };

const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const maximumBytes = 2 * 1024 * 1024;

class PhotoResponseError extends Error {}

async function readPhotoResponse(response: Response): Promise<AdminProfilePhotoState> {
  const result = await response.json();
  if (!response.ok) {
    throw new PhotoResponseError(
      typeof result.error === "string"
        ? result.error
        : "The profile photo could not be updated. Please try again.",
    );
  }
  return result as AdminProfilePhotoState;
}

export function AdminProfilePhoto({ profileId }: { profileId: string }) {
  const router = useRouter();
  const fieldId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);
  const [state, setState] = useState<AdminProfilePhotoState | null>(null);
  const [selected, setSelected] = useState<SelectedPhoto | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [reloadCount, setReloadCount] = useState(0);
  const [failedPreview, setFailedPreview] = useState<string | null>(null);
  const endpoint = `/api/admin/profile-photo?profileId=${encodeURIComponent(profileId)}`;

  useEffect(() => {
    const controller = new AbortController();
    const version = ++requestVersion.current;
    requestRef.current = controller;

    async function load() {
      try {
        const response = await fetch(endpoint, {
          cache: "no-store",
          credentials: "same-origin",
          signal: controller.signal,
        });
        const result = await readPhotoResponse(response);
        if (version !== requestVersion.current || controller.signal.aborted) return;
        setState(result);
      } catch {
        if (version !== requestVersion.current || controller.signal.aborted) return;
        setFeedback({
          message:
            "The photo controls could not be loaded. They are available for the primary Profile & About record only. Retry after confirming you are still signed in.",
          error: true,
        });
      } finally {
        if (version === requestVersion.current && !controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      requestRef.current?.abort();
    };
  }, [endpoint, reloadCount]);

  useEffect(() => {
    if (!selected) return;
    return () => URL.revokeObjectURL(selected.preview);
  }, [selected]);

  function clearSelection() {
    setSelected(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function choosePhoto(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setFeedback(null);
    if (!file) return;

    if (!allowedTypes.has(file.type) || file.size === 0 || file.size > maximumBytes) {
      clearSelection();
      setFeedback({
        message: "Choose a JPEG, PNG, or WebP image no larger than 2 MiB. Animated images are not supported.",
        error: true,
      });
      return;
    }

    setSelected({ file, preview: URL.createObjectURL(file) });
  }

  async function savePhoto(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!state?.storageConfigured || !selected || pending) return;
    await mutate("PUT", selected.file);
  }

  async function removePhoto() {
    if (!state?.storageConfigured || !state.avatarId || pending) return;
    if (!window.confirm("Remove the current profile photo? The public portfolio will return to its no-photo layout. Profile text and other content will not change.")) {
      return;
    }
    await mutate("DELETE");
  }

  async function mutate(method: "PUT" | "DELETE", file?: File) {
    if (!state) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    const version = ++requestVersion.current;
    requestRef.current = controller;
    setPending(true);
    setFeedback(null);

    try {
      const response = await fetch(endpoint, {
        method,
        cache: "no-store",
        credentials: "same-origin",
        signal: controller.signal,
        headers: {
          "x-profile-photo-current": state.avatarId ?? "none",
          ...(file ? { "Content-Type": file.type } : {}),
          ...(method === "DELETE" ? { "x-profile-photo-remove-confirm": "true" } : {}),
        },
        ...(file ? { body: file } : {}),
      });
      const result = await readPhotoResponse(response);
      if (version !== requestVersion.current || controller.signal.aborted) return;
      setState(result);
      clearSelection();
      setFeedback({
        message: method === "DELETE"
          ? "Profile photo removed. The public portfolio will use its no-photo layout."
          : "Profile photo saved. It appears publicly when this profile is published.",
        error: false,
      });
      router.refresh();
    } catch (error) {
      if (version !== requestVersion.current || controller.signal.aborted) return;
      setFeedback({
        message: error instanceof PhotoResponseError
          ? error.message
          : "The photo request could not be completed. Reload the photo controls to check the saved state before trying again.",
        error: true,
      });
    } finally {
      if (version === requestVersion.current && !controller.signal.aborted) {
        setPending(false);
      }
    }
  }

  function reloadPhoto() {
    if (pending) return;
    clearSelection();
    setState(null);
    setFeedback(null);
    setFailedPreview(null);
    setLoading(true);
    setReloadCount((current) => current + 1);
  }

  const preview = selected?.preview ?? state?.photo?.src;
  const disabled = loading || pending || !state?.storageConfigured;

  return (
    <section aria-labelledby={`${fieldId}-title`} aria-busy={loading || pending}>
      <div className="flex items-center gap-2 font-mono text-[0.68rem] font-semibold tracking-[0.15em] text-primary uppercase">
        <Camera aria-hidden="true" className="size-4" />
        Profile &amp; About
      </div>
      <h2 id={`${fieldId}-title`} className="mt-3 font-serif text-3xl font-medium tracking-[-0.035em] text-ink">
        Profile photo
      </h2>
      <p id={`${fieldId}-description`} className="mt-3 max-w-3xl text-sm leading-7 text-muted">
        Upload one professional portrait for Home and About. Photo changes are saved separately and do not change profile text or visibility. Only a published primary profile displays its photo publicly.
      </p>

      <form onSubmit={savePhoto} className="mt-6 rounded-2xl border border-line bg-surface p-5 sm:p-6">
        <div className="grid gap-6 sm:grid-cols-[10rem_minmax(0,1fr)]">
          <div>
            <div className="flex aspect-square items-center justify-center overflow-hidden rounded-2xl border border-line bg-canvas">
              {preview && failedPreview !== preview ? (
                <Image
                  key={preview}
                  src={preview}
                  alt={selected ? "Selected profile photo preview, not yet saved" : state?.photo?.alt ?? "Current profile photo"}
                  width={selected ? 320 : state?.photo?.width ?? 320}
                  height={selected ? 320 : state?.photo?.height ?? 320}
                  unoptimized
                  className="h-full w-full object-contain"
                  onError={() => setFailedPreview(preview)}
                />
              ) : (
                <p className="px-4 text-center text-xs leading-6 text-muted">
                  {loading ? "Loading photo…" : preview ? "Preview unavailable" : "No profile photo"}
                </p>
              )}
            </div>
            <p className="mt-2 text-center text-xs text-muted">
              {selected ? "Selected · not saved" : state?.avatarId ? "Current photo" : "No-photo layout"}
            </p>
          </div>

          <div className="min-w-0">
            {state && !state.storageConfigured ? (
              <p className="mb-4 rounded-xl border border-line bg-canvas p-4 text-sm leading-6 text-muted">
                Private image storage has not been configured yet. Upload, replacement, and removal are unavailable until setup is complete. Existing profile content is unchanged.
              </p>
            ) : null}
            <label htmlFor={fieldId} className="block text-sm font-semibold text-ink">
              {state?.avatarId ? "Choose a replacement photo" : "Choose a profile photo"}
            </label>
            <input
              ref={inputRef}
              id={fieldId}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              disabled={disabled}
              onChange={choosePhoto}
              aria-describedby={`${fieldId}-limits ${fieldId}-description`}
              className="mt-2 block min-h-11 w-full min-w-0 rounded-xl border border-line bg-canvas p-2 text-sm text-ink file:mr-3 file:rounded-lg file:border-0 file:bg-primary-soft file:px-3 file:py-2 file:font-semibold file:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-55"
            />
            <p id={`${fieldId}-limits`} className="mt-2 text-xs leading-6 text-muted">
              JPEG, PNG, or WebP · maximum 2 MiB · no animation. Images are checked, resized, and optimized securely before saving.
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <Button type="submit" disabled={disabled || !selected}>
                <Save aria-hidden="true" className="size-4" />
                {pending ? "Saving…" : "Save photo"}
              </Button>
              {selected ? (
                <Button variant="ghost" disabled={pending} onClick={() => { clearSelection(); setFeedback(null); }}>
                  Cancel selection
                </Button>
              ) : state?.avatarId ? (
                <Button variant="outline" disabled={disabled} onClick={removePhoto}>
                  <Trash2 aria-hidden="true" className="size-4" />
                  Remove photo
                </Button>
              ) : null}
              {!loading && !pending ? (
                <Button variant="ghost" onClick={reloadPhoto}>
                  Reload photo controls
                </Button>
              ) : null}
            </div>
          </div>
        </div>
        <div aria-live="polite" aria-atomic="true">
          {feedback ? (
            <p
              role={feedback.error ? "alert" : "status"}
              className={cn(
                "mt-5 rounded-xl border px-4 py-3 text-sm leading-6",
                feedback.error
                  ? "border-danger/30 bg-danger/[0.06] text-danger"
                  : "border-secondary/30 bg-secondary-soft text-secondary",
              )}
            >
              {feedback.message}
            </p>
          ) : null}
        </div>
      </form>
    </section>
  );
}
