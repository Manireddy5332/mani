const monthNames = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

export function formatMonthYear(
  year: number | null,
  month: number | null,
): string | null {
  if (year === null) return null;
  if (month === null) return String(year);
  const monthName = monthNames[month - 1];
  return monthName ? `${monthName} ${year}` : String(year);
}

export function formatPartialPeriod(period: Readonly<{
  startYear: number | null;
  startMonth: number | null;
  endYear: number | null;
  endMonth: number | null;
  isCurrent: boolean;
}>): string | null {
  const start = formatMonthYear(period.startYear, period.startMonth);
  const end = period.isCurrent
    ? "Present"
    : formatMonthYear(period.endYear, period.endMonth);

  if (start && end) return `${start} — ${end}`;
  return start ?? end;
}

