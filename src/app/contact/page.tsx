import { notFound } from "next/navigation";

import { ContactPage } from "@/features/contact";
import { getPublicContactPageData } from "@/features/contact/queries.server";
import { createPublicPageMetadata } from "@/lib/metadata";

const description =
  "Contact Manikanta Reddy Anugu by email or LinkedIn for academic, research, and professional AI/ML conversations.";

export const metadata = createPublicPageMetadata({
  title: "Contact",
  description,
  path: "/contact",
});

export const dynamic = "force-dynamic";

export default async function ContactRoute() {
  const contact = await getPublicContactPageData();
  if (!contact) notFound();
  return <ContactPage contact={contact} />;
}
