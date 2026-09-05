import { Braces, Info } from "lucide-react";

import { AdminProjectNestedEditor } from "@/components/admin/admin-project-nested-editor";
import { AdminResearchQuestionsEditor } from "@/components/admin/admin-research-questions-editor";
import type { AdminNestedContentDto } from "@/features/admin/types";

type AdminNestedContentEditorProps = {
  content: AdminNestedContentDto;
};

export function AdminNestedContentEditor({
  content,
}: AdminNestedContentEditorProps) {
  return (
    <section aria-labelledby="nested-content-title">
      <div className="border-b border-line pb-6">
        <div className="flex items-center gap-2 font-mono text-[0.68rem] font-semibold tracking-[0.15em] text-primary uppercase">
          <Braces aria-hidden="true" className="size-4" />
          Structured content
        </div>
        <h2
          id="nested-content-title"
          className="mt-3 font-serif text-3xl font-medium tracking-[-0.035em] text-ink sm:text-4xl"
        >
          Supporting details
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-muted">
          Manage ordered details that belong specifically to {content.parent.title}.
          These entries inherit the parent record&apos;s visibility and are not published independently.
        </p>
      </div>

      {content.parent.status === "ARCHIVED" ? (
        <div className="mt-6 flex gap-3 rounded-2xl border border-secondary/30 bg-secondary-soft p-4 text-sm leading-6 text-ink">
          <Info aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-secondary" />
          <p>
            The parent record is archived. You can prepare these details here, but they remain private until the parent is restored and published.
          </p>
        </div>
      ) : null}

      <div className="mt-7">
        {content.resource === "research-projects" ? (
          <AdminResearchQuestionsEditor content={content} />
        ) : (
          <AdminProjectNestedEditor content={content} />
        )}
      </div>
    </section>
  );
}
