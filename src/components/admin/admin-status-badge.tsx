import { Badge } from "@/components/ui";
import { cn } from "@/lib/cn";

type AdminStatusBadgeProps = {
  status?: string;
};

const labels: Record<string, string> = {
  ARCHIVED: "Archived",
  DRAFT: "Draft",
  PUBLISHED: "Published",
};

export function AdminStatusBadge({ status }: AdminStatusBadgeProps) {
  if (!status) {
    return <span className="text-xs text-muted">Not applicable</span>;
  }

  return (
    <Badge
      variant={status === "PUBLISHED" ? "accent" : "outline"}
      className={cn(status === "ARCHIVED" && "opacity-65")}
    >
      {labels[status] ?? status.replaceAll("_", " ").toLowerCase()}
    </Badge>
  );
}
