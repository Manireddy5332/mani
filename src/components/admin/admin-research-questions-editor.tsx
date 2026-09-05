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
  AdminResearchNestedContentDto,
  AdminResearchNestedContentInput,
} from "@/features/admin/types";
import { cn } from "@/lib/cn";

type DraftQuestion = {
  clientKey: string;
  id?: string;
  key: string;
  question: string;
};

type Feedback = {
  message: string;
  tone: "error" | "info" | "success";
};

function createDraftQuestions(
  content: AdminResearchNestedContentDto,
): DraftQuestion[] {
  return content.questions.map((question) => ({
    clientKey: question.id,
    id: question.id,
    key: question.key,
    question: question.question,
  }));
}

type AdminResearchQuestionsEditorProps = {
  content: AdminResearchNestedContentDto;
};

export function AdminResearchQuestionsEditor({
  content,
}: AdminResearchQuestionsEditorProps) {
  const router = useRouter();
  const generatedId = useId().replaceAll(":", "");
  const addButtonId = `${generatedId}-add-question`;
  const formRef = useRef<HTMLFormElement>(null);
  const nextClientId = useRef(0);
  const [questions, setQuestions] = useState(() => createDraftQuestions(content));
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

  function addQuestion() {
    const clientKey = `new-question-${++nextClientId.current}`;
    setQuestions((current) => [
      ...current,
      { clientKey, key: "", question: "" },
    ]);
    markDirty();
    setAnnouncement(`Added research question ${questions.length + 1}.`);
  }

  function updateQuestion(
    index: number,
    field: "key" | "question",
    value: string,
  ) {
    setQuestions((current) =>
      current.map((question, questionIndex) =>
        questionIndex === index ? { ...question, [field]: value } : question,
      ),
    );
    markDirty();
  }

  function removeQuestion(index: number) {
    const question = questions[index];
    if (!question) return;

    setQuestions((current) =>
      current.filter((_, questionIndex) => questionIndex !== index),
    );
    if (question.id) {
      setRemovedIds((current) =>
        current.includes(question.id as string)
          ? current
          : [...current, question.id as string],
      );
    }
    markDirty();
    setAnnouncement(
      `Removed ${question.key.trim() || `research question ${index + 1}`} from the draft.`,
    );
    focusNestedAddButton(addButtonId);
  }

  function moveQuestion(index: number, direction: -1 | 1) {
    const question = questions[index];
    if (!question) return;

    setQuestions((current) => moveNestedItem(current, index, direction));
    markDirty();
    setAnnouncement(
      `${question.key.trim() || `Research question ${index + 1}`} moved to position ${index + direction + 1}.`,
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending || !dirty) return;

    const payload: AdminResearchNestedContentInput = {
      revision,
      removedIds,
      questions: questions.map(({ id, key, question }) => ({
        ...(id ? { id } : {}),
        key,
        question,
      })),
    };

    setPending(true);
    setFeedback(undefined);
    setFieldErrors(undefined);

    try {
      let result = await saveAdminNestedContentAction(
        "research-projects",
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
          "research-projects",
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

      if (result.content?.resource !== "research-projects") {
        setFeedback({
          message: "The changes were saved, but the editor could not refresh them. Reload this page before making another change.",
          tone: "error",
        });
        router.refresh();
        return;
      }

      setQuestions(createDraftQuestions(result.content));
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
          addButtonId={addButtonId}
          title="Research questions"
          description="Connect this research direction to explicit, verified questions. The order here controls their public reading order when the parent project is published."
          addLabel="Add question"
          onAdd={addQuestion}
          pending={pending}
        />

        {questions.length === 0 ? (
          <div className="mt-6 rounded-xl border border-dashed border-line bg-ink/[0.02] px-5 py-8 text-center">
            <p className="text-sm font-semibold text-ink">No research questions</p>
            <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted">
              Leave this collection empty until a research question is verified, or add one with the button above.
            </p>
          </div>
        ) : (
          <div className="mt-6 space-y-4">
            {questions.map((question, index) => {
              const fieldId = `${generatedId}-question-${question.clientKey}`;

              return (
                <fieldset
                  key={question.clientKey}
                  disabled={pending}
                  className="rounded-xl border border-line bg-canvas p-4 sm:p-5"
                >
                  <legend className="sr-only">Research question {index + 1}</legend>
                  <NestedItemHeader
                    index={index}
                    itemLabel="Question"
                    title={question.key.trim() || "Untitled question"}
                    total={questions.length}
                    pending={pending}
                    onMove={(direction) => moveQuestion(index, direction)}
                    onRemove={() => removeQuestion(index)}
                  />
                  <div className="mt-5 grid gap-5 md:grid-cols-[minmax(11rem,0.42fr)_minmax(0,1fr)]">
                    <NestedTextField
                      id={`${fieldId}-key`}
                      label="Internal key"
                      value={question.key}
                      required
                      slug
                      placeholder="evaluation-reliability"
                      description="Lowercase words separated by hyphens; unique within this research project."
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `questions.${index}.key`,
                      )}
                      onChange={(value) => updateQuestion(index, "key", value)}
                    />
                    <NestedTextField
                      id={`${fieldId}-question`}
                      label="Research question"
                      value={question.question}
                      required
                      multiline
                      rows={4}
                      errors={getNestedFieldErrors(
                        fieldErrors,
                        `questions.${index}.question`,
                      )}
                      onChange={(value) =>
                        updateQuestion(index, "question", value)
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
