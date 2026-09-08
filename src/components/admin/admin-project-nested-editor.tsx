"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent } from "react";

import {
  NestedCollectionHeader,
  NestedItemHeader,
  NestedSaveBar,
  NestedTextField,
  confirmNestedRemoval,
  focusNestedAddButton,
  getNestedFieldErrors,
  getNestedSummaryErrors,
  moveNestedItem,
} from "@/components/admin/admin-nested-editor-ui";
import { saveAdminNestedContentAction } from "@/features/admin/actions";
import type {
  AdminProjectNestedContentDto,
  AdminProjectNestedContentInput,
} from "@/features/admin/types";
import { cn } from "@/lib/cn";

type DraftContribution = {
  clientKey: string;
  description: string;
  id?: string;
  key: string;
  label: string;
};

type DraftFeature = {
  clientKey: string;
  description: string;
  id?: string;
  key: string;
  name: string;
};

type DraftTechnology = {
  clientKey: string;
  id?: string;
  name: string;
  slug: string;
};

type Feedback = {
  message: string;
  tone: "error" | "info" | "success";
};

function createDraftProjectContent(content: AdminProjectNestedContentDto) {
  return {
    contributions: content.contributions.map(
      (item): DraftContribution => ({
        clientKey: item.id,
        description: item.description,
        id: item.id,
        key: item.key,
        label: item.label,
      }),
    ),
    features: content.features.map(
      (item): DraftFeature => ({
        clientKey: item.id,
        description: item.description ?? "",
        id: item.id,
        key: item.key,
        name: item.name,
      }),
    ),
    technologies: content.technologies.map(
      (item): DraftTechnology => ({
        clientKey: item.id,
        id: item.id,
        name: item.name,
        slug: item.slug,
      }),
    ),
  };
}

type AdminProjectNestedEditorProps = {
  content: AdminProjectNestedContentDto;
};

export function AdminProjectNestedEditor({
  content,
}: AdminProjectNestedEditorProps) {
  const router = useRouter();
  const generatedId = useId().replaceAll(":", "");
  const addContributionButtonId = `${generatedId}-add-contribution`;
  const addFeatureButtonId = `${generatedId}-add-feature`;
  const addTechnologyButtonId = `${generatedId}-add-technology`;
  const formRef = useRef<HTMLFormElement>(null);
  const nextClientId = useRef(0);
  const initial = createDraftProjectContent(content);
  const [contributions, setContributions] = useState(initial.contributions);
  const [features, setFeatures] = useState(initial.features);
  const [technologies, setTechnologies] = useState(initial.technologies);
  const [revision, setRevision] = useState(content.parent.updatedAt);
  const [removedIds, setRemovedIds] = useState<string[]>([]);
  const [dirty, setDirty] = useState(false);
  const [pending, setPending] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string[]>>();
  const [feedback, setFeedback] = useState<Feedback>();
  const [announcement, setAnnouncement] = useState("");
  const summaryErrors = getNestedSummaryErrors(fieldErrors);

  useEffect(() => {
    if (!fieldErrors) return;

    formRef.current
      ?.querySelector<HTMLElement>('[aria-invalid="true"]')
      ?.focus();
  }, [fieldErrors]);

  function markDirty() {
    setDirty(true);
    setFeedback(undefined);
    setFieldErrors(undefined);
  }

  function trackRemoval(id: string | undefined) {
    if (!id) return;
    setRemovedIds((current) =>
      current.includes(id) ? current : [...current, id],
    );
  }

  function addContribution() {
    const clientKey = `new-contribution-${++nextClientId.current}`;
    setContributions((current) => [
      ...current,
      { clientKey, description: "", key: "", label: "" },
    ]);
    markDirty();
    setAnnouncement(`Added contribution ${contributions.length + 1}.`);
  }

  function updateContribution(
    index: number,
    field: "description" | "key" | "label",
    value: string,
  ) {
    setContributions((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, [field]: value } : item,
      ),
    );
    markDirty();
  }

  function removeContribution(index: number) {
    const item = contributions[index];
    if (!item) return;
    setContributions((current) =>
      current.filter((_, itemIndex) => itemIndex !== index),
    );
    trackRemoval(item.id);
    markDirty();
    setAnnouncement(
      `Removed ${item.label.trim() || `contribution ${index + 1}`} from the draft.`,
    );
    focusNestedAddButton(addContributionButtonId);
  }

  function moveContribution(index: number, direction: -1 | 1) {
    const item = contributions[index];
    if (!item) return;
    setContributions((current) => moveNestedItem(current, index, direction));
    markDirty();
    setAnnouncement(
      `${item.label.trim() || `Contribution ${index + 1}`} moved to position ${index + direction + 1}.`,
    );
  }

  function addFeature() {
    const clientKey = `new-feature-${++nextClientId.current}`;
    setFeatures((current) => [
      ...current,
      { clientKey, description: "", key: "", name: "" },
    ]);
    markDirty();
    setAnnouncement(`Added feature ${features.length + 1}.`);
  }

  function updateFeature(
    index: number,
    field: "description" | "key" | "name",
    value: string,
  ) {
    setFeatures((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, [field]: value } : item,
      ),
    );
    markDirty();
  }

  function removeFeature(index: number) {
    const item = features[index];
    if (!item) return;
    setFeatures((current) =>
      current.filter((_, itemIndex) => itemIndex !== index),
    );
    trackRemoval(item.id);
    markDirty();
    setAnnouncement(
      `Removed ${item.name.trim() || `feature ${index + 1}`} from the draft.`,
    );
    focusNestedAddButton(addFeatureButtonId);
  }

  function moveFeature(index: number, direction: -1 | 1) {
    const item = features[index];
    if (!item) return;
    setFeatures((current) => moveNestedItem(current, index, direction));
    markDirty();
    setAnnouncement(
      `${item.name.trim() || `Feature ${index + 1}`} moved to position ${index + direction + 1}.`,
    );
  }

  function addTechnology() {
    const clientKey = `new-technology-${++nextClientId.current}`;
    setTechnologies((current) => [
      ...current,
      { clientKey, name: "", slug: "" },
    ]);
    markDirty();
    setAnnouncement(`Added technology ${technologies.length + 1}.`);
  }

  function updateTechnology(
    index: number,
    field: "name" | "slug",
    value: string,
  ) {
    setTechnologies((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, [field]: value } : item,
      ),
    );
    markDirty();
  }

  function removeTechnology(index: number) {
    const item = technologies[index];
    if (!item) return;
    setTechnologies((current) =>
      current.filter((_, itemIndex) => itemIndex !== index),
    );
    trackRemoval(item.id);
    markDirty();
    setAnnouncement(
      `Removed ${item.name.trim() || `technology ${index + 1}`} from the draft.`,
    );
    focusNestedAddButton(addTechnologyButtonId);
  }

  function moveTechnology(index: number, direction: -1 | 1) {
    const item = technologies[index];
    if (!item) return;
    setTechnologies((current) => moveNestedItem(current, index, direction));
    markDirty();
    setAnnouncement(
      `${item.name.trim() || `Technology ${index + 1}`} moved to position ${index + direction + 1}.`,
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending || !dirty) return;

    const payload: AdminProjectNestedContentInput = {
      revision,
      removedIds,
      contributions: contributions.map(({ id, key, label, description }) => ({
        ...(id ? { id } : {}),
        key,
        label,
        description,
      })),
      features: features.map(({ id, key, name, description }) => ({
        ...(id ? { id } : {}),
        key,
        name,
        description,
      })),
      technologies: technologies.map(({ id, slug, name }) => ({
        ...(id ? { id } : {}),
        slug,
        name,
      })),
    };

    setPending(true);
    setFeedback(undefined);
    setFieldErrors(undefined);

    try {
      let result = await saveAdminNestedContentAction(
        "projects",
        content.parent.id,
        payload,
      );

      if (!result.ok && result.confirmation) {
        if (!confirmNestedRemoval(result.confirmation.labels)) {
          setFeedback({
            message: "Removal cancelled. No changes were saved.",
            tone: "info",
          });
          return;
        }

        result = await saveAdminNestedContentAction(
          "projects",
          content.parent.id,
          {
            ...payload,
            removedIds: result.confirmation.removedIds,
            confirmRemoval: true,
          },
        );
      }

      if (!result.ok) {
        setFieldErrors(result.fieldErrors);
        setFeedback({ message: result.message, tone: "error" });
        return;
      }

      if (result.content?.resource !== "projects") {
        setFeedback({
          message: "The changes were saved, but the editor could not refresh them. Reload this page before making another change.",
          tone: "error",
        });
        router.refresh();
        return;
      }

      const next = createDraftProjectContent(result.content);
      setContributions(next.contributions);
      setFeatures(next.features);
      setTechnologies(next.technologies);
      setRevision(result.content.parent.updatedAt);
      setRemovedIds([]);
      setDirty(false);
      setFeedback({ message: result.message, tone: "success" });
      router.refresh();
    } catch {
      setFeedback({
        message: "The nested content could not be saved. Please try again.",
        tone: "error",
      });
    } finally {
      setPending(false);
    }
  }

  return (
    <form
      ref={formRef}
      onSubmit={handleSubmit}
      aria-busy={pending}
      className="space-y-6"
    >
      <div className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
        <NestedCollectionHeader
          addButtonId={addContributionButtonId}
          title="Contributions"
          description="Describe verified areas of contribution without adding unsupported ownership, metrics, or outcomes. The displayed order follows this list."
          addLabel="Add contribution"
          onAdd={addContribution}
          pending={pending}
        />
        {contributions.length === 0 ? (
          <EmptyCollection label="contributions" />
        ) : (
          <div className="mt-6 space-y-4">
            {contributions.map((item, index) => {
              const fieldId = `${generatedId}-contribution-${item.clientKey}`;
              return (
                <fieldset
                  key={item.clientKey}
                  disabled={pending}
                  className="rounded-xl border border-line bg-canvas p-4 sm:p-5"
                >
                  <legend className="sr-only">Contribution {index + 1}</legend>
                  <NestedItemHeader
                    index={index}
                    itemLabel="Contribution"
                    title={item.label.trim() || "Untitled contribution"}
                    total={contributions.length}
                    pending={pending}
                    onMove={(direction) => moveContribution(index, direction)}
                    onRemove={() => removeContribution(index)}
                  />
                  <div className="mt-5 grid gap-5 md:grid-cols-2">
                    <NestedTextField
                      id={`${fieldId}-key`}
                      label="Internal key"
                      value={item.key}
                      required
                      slug
                      placeholder="model-development"
                      description="Lowercase words separated by hyphens; unique within this project."
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `contributions.${index}.key`,
                      )}
                      onChange={(value) =>
                        updateContribution(index, "key", value)
                      }
                    />
                    <NestedTextField
                      id={`${fieldId}-label`}
                      label="Label"
                      value={item.label}
                      required
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `contributions.${index}.label`,
                      )}
                      onChange={(value) =>
                        updateContribution(index, "label", value)
                      }
                    />
                    <div className="md:col-span-2">
                      <NestedTextField
                        id={`${fieldId}-description`}
                        label="Description"
                        value={item.description}
                        required
                        multiline
                        rows={4}
                        errors={getNestedFieldErrors(
                          fieldErrors,
                          `contributions.${index}.description`,
                        )}
                        onChange={(value) =>
                          updateContribution(index, "description", value)
                        }
                      />
                    </div>
                  </div>
                </fieldset>
              );
            })}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
        <NestedCollectionHeader
          addButtonId={addFeatureButtonId}
          title="Features"
          description="Maintain verified project capabilities as concise, ordered feature records. Optional descriptions can stay blank."
          addLabel="Add feature"
          onAdd={addFeature}
          pending={pending}
        />
        {features.length === 0 ? (
          <EmptyCollection label="features" />
        ) : (
          <div className="mt-6 space-y-4">
            {features.map((item, index) => {
              const fieldId = `${generatedId}-feature-${item.clientKey}`;
              return (
                <fieldset
                  key={item.clientKey}
                  disabled={pending}
                  className="rounded-xl border border-line bg-canvas p-4 sm:p-5"
                >
                  <legend className="sr-only">Feature {index + 1}</legend>
                  <NestedItemHeader
                    index={index}
                    itemLabel="Feature"
                    title={item.name.trim() || "Untitled feature"}
                    total={features.length}
                    pending={pending}
                    onMove={(direction) => moveFeature(index, direction)}
                    onRemove={() => removeFeature(index)}
                  />
                  <div className="mt-5 grid gap-5 md:grid-cols-2">
                    <NestedTextField
                      id={`${fieldId}-key`}
                      label="Internal key"
                      value={item.key}
                      required
                      slug
                      placeholder="evidence-traceability"
                      description="Lowercase words separated by hyphens; unique within this project."
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `features.${index}.key`,
                      )}
                      onChange={(value) => updateFeature(index, "key", value)}
                    />
                    <NestedTextField
                      id={`${fieldId}-name`}
                      label="Name"
                      value={item.name}
                      required
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `features.${index}.name`,
                      )}
                      onChange={(value) => updateFeature(index, "name", value)}
                    />
                    <div className="md:col-span-2">
                      <NestedTextField
                        id={`${fieldId}-description`}
                        label="Description"
                        value={item.description}
                        multiline
                        rows={3}
                        errors={getNestedFieldErrors(
                          fieldErrors,
                          `features.${index}.description`,
                        )}
                        onChange={(value) =>
                          updateFeature(index, "description", value)
                        }
                      />
                    </div>
                  </div>
                </fieldset>
              );
            })}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
        <NestedCollectionHeader
          addButtonId={addTechnologyButtonId}
          title="Technologies"
          description="List only technologies verified for this project. Their order controls how the stack is presented publicly."
          addLabel="Add technology"
          onAdd={addTechnology}
          pending={pending}
        />
        {technologies.length === 0 ? (
          <EmptyCollection label="technologies" />
        ) : (
          <div className="mt-6 grid gap-4 xl:grid-cols-2">
            {technologies.map((item, index) => {
              const fieldId = `${generatedId}-technology-${item.clientKey}`;
              return (
                <fieldset
                  key={item.clientKey}
                  disabled={pending}
                  className="rounded-xl border border-line bg-canvas p-4 sm:p-5"
                >
                  <legend className="sr-only">Technology {index + 1}</legend>
                  <NestedItemHeader
                    index={index}
                    itemLabel="Technology"
                    title={item.name.trim() || "Untitled technology"}
                    total={technologies.length}
                    pending={pending}
                    onMove={(direction) => moveTechnology(index, direction)}
                    onRemove={() => removeTechnology(index)}
                  />
                  <div className="mt-5 grid gap-5 sm:grid-cols-2">
                    <NestedTextField
                      id={`${fieldId}-slug`}
                      label="Slug"
                      value={item.slug}
                      required
                      slug
                      placeholder="pytorch"
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `technologies.${index}.slug`,
                      )}
                      onChange={(value) =>
                        updateTechnology(index, "slug", value)
                      }
                    />
                    <NestedTextField
                      id={`${fieldId}-name`}
                      label="Display name"
                      value={item.name}
                      required
                      placeholder="PyTorch"
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `technologies.${index}.name`,
                      )}
                      onChange={(value) =>
                        updateTechnology(index, "name", value)
                      }
                    />
                  </div>
                </fieldset>
              );
            })}
          </div>
        )}
      </div>

      <p className="sr-only" aria-live="polite">
        {announcement}
      </p>

      {feedback ? (
        <div
          aria-live="polite"
          role={feedback.tone === "error" ? "alert" : "status"}
          className={cn(
            "rounded-xl border px-4 py-3 text-sm font-medium",
            feedback.tone === "success" &&
              "border-secondary/30 bg-secondary-soft text-secondary",
            feedback.tone === "error" &&
              "border-danger/30 bg-danger/[0.06] text-danger",
            feedback.tone === "info" && "border-line bg-ink/[0.03] text-ink",
          )}
        >
          {feedback.message}
          {summaryErrors.length ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 font-normal">
              {summaryErrors.map((message) => (
                <li key={message}>{message}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      <NestedSaveBar
        dirty={dirty}
        pending={pending}
        removedCount={removedIds.length}
      />
    </form>
  );
}

function EmptyCollection({ label }: { label: string }) {
  return (
    <div className="mt-6 rounded-xl border border-dashed border-line bg-ink/[0.02] px-5 py-8 text-center">
      <p className="text-sm font-semibold text-ink">No {label}</p>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted">
        Leave this collection empty until verified information is available, or add an item with the button above.
      </p>
    </div>
  );
}
