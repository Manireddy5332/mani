import type { LucideIcon } from "lucide-react";
import {
  Award,
  BookOpen,
  BriefcaseBusiness,
  FileText,
  FlaskConical,
  FolderKanban,
  GraduationCap,
  Link2,
  Newspaper,
  Settings2,
  UserRound,
  Wrench,
} from "lucide-react";

import type { AdminResourceDefinition } from "@/features/admin/types";

const icons: Record<AdminResourceDefinition["icon"], LucideIcon> = {
  article: Newspaper,
  award: Award,
  briefcase: BriefcaseBusiness,
  education: GraduationCap,
  folder: FolderKanban,
  link: Link2,
  profile: UserRound,
  publication: BookOpen,
  research: FlaskConical,
  resume: FileText,
  settings: Settings2,
  skill: Wrench,
};

type AdminIconProps = {
  className?: string;
  name: AdminResourceDefinition["icon"];
};

export function AdminIcon({ className, name }: AdminIconProps) {
  const Icon = icons[name];
  return <Icon aria-hidden="true" className={className} />;
}
