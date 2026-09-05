import { ArrowLeft, FlaskConical } from "lucide-react";

import { PageIntro } from "@/components/site/page-intro";
import { ButtonLink, Surface } from "@/components/ui";

export default function NotFound() {
  return (
    <main id="main-content" className="flex-1">
      <PageIntro
        eyebrow="Page not found"
        title="That page is not available."
        description="The requested address may be incorrect, or it may refer to material that is not currently available as a public page."
        actions={
          <>
            <ButtonLink href="/">
              <ArrowLeft aria-hidden="true" className="size-4" />
              Return home
            </ButtonLink>
            <ButtonLink href="/research" variant="outline">
              <FlaskConical aria-hidden="true" className="size-4" />
              Explore research
            </ButtonLink>
          </>
        }
        aside={
          <Surface padding="lg" variant="accent">
            <p className="eyebrow text-primary">Current public areas</p>
            <p className="mt-5 text-sm leading-7 text-muted">
              Research, projects, experience, writing, about, resume, and
              contact pages are available from the site navigation.
            </p>
          </Surface>
        }
      />
    </main>
  );
}
