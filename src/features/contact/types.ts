export type ContactMethod = {
  readonly key: string;
  readonly label: string;
  readonly value: string;
  readonly href?: string;
  readonly external?: boolean;
  readonly description: string;
  readonly kind:
    | "email"
    | "linkedin"
    | "github"
    | "website"
    | "other"
    | "location"
    | "repository";
};

export type ContactPageData = {
  readonly name: string;
  readonly introduction: string;
  readonly methods: readonly ContactMethod[];
};
