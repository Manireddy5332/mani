export type PublicProfilePhoto = {
  src: string;
  width: number;
  height: number;
  alt: string;
};

export type AdminProfilePhotoState = {
  profileId: string;
  avatarId: string | null;
  photo: PublicProfilePhoto | null;
  storageConfigured: boolean;
};
