-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateEnum
CREATE TYPE "ContentStatus" AS ENUM ('DRAFT', 'PUBLISHED', 'ARCHIVED');

-- CreateEnum
CREATE TYPE "ContentSource" AS ENUM ('ACADEMIC_CV', 'USER_PROVIDED', 'ADMIN');

-- CreateEnum
CREATE TYPE "SocialLinkKind" AS ENUM ('EMAIL', 'LINKEDIN', 'GITHUB', 'WEBSITE', 'OTHER');

-- CreateEnum
CREATE TYPE "ResearchStage" AS ENUM ('RESEARCH_INTEREST', 'EXPLORING', 'EARLY_STAGE', 'IN_PROGRESS', 'WORKING_PAPER', 'SUBMITTED', 'ACCEPTED', 'PUBLISHED', 'ARCHIVED');

-- CreateEnum
CREATE TYPE "PublicationStage" AS ENUM ('WORKING_PAPER', 'SUBMITTED', 'ACCEPTED', 'PUBLISHED', 'ARCHIVED');

-- CreateEnum
CREATE TYPE "PublicationType" AS ENUM ('JOURNAL_ARTICLE', 'CONFERENCE_PAPER', 'WORKSHOP_PAPER', 'PREPRINT', 'BOOK_CHAPTER', 'THESIS', 'OTHER');

-- CreateEnum
CREATE TYPE "ProjectStatus" AS ENUM ('PLANNED', 'IN_PROGRESS', 'ON_HOLD', 'COMPLETED', 'ARCHIVED');

-- CreateEnum
CREATE TYPE "ExperienceType" AS ENUM ('CLIENT_ENGAGEMENT', 'EMPLOYMENT', 'INTERNSHIP', 'CONTRACT', 'OTHER');

-- CreateEnum
CREATE TYPE "MediaRole" AS ENUM ('COVER', 'GALLERY', 'ARCHITECTURE_DIAGRAM', 'FIGURE', 'FEATURED_IMAGE', 'INLINE', 'ATTACHMENT');

-- CreateEnum
CREATE TYPE "ContactSubmissionStatus" AS ENUM ('NEW', 'READ', 'REPLIED', 'ARCHIVED', 'SPAM');

-- CreateTable
CREATE TABLE "Profile" (
    "id" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "positioning" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "headline" TEXT NOT NULL,
    "introduction" TEXT NOT NULL,
    "about" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "location" TEXT,
    "avatarId" UUID,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Profile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SocialLink" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "kind" "SocialLinkKind" NOT NULL,
    "label" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "handle" TEXT,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "SocialLink_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ResearchInterest" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ResearchInterest_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ResearchProject" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "summary" TEXT NOT NULL,
    "abstract" TEXT,
    "motivation" TEXT,
    "methodology" TEXT,
    "methodologySummary" TEXT,
    "domain" TEXT,
    "researchArea" TEXT,
    "format" TEXT,
    "scopeBoundary" TEXT,
    "evidenceStatus" TEXT,
    "advisor" TEXT,
    "collaborators" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "technologies" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "datasets" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "githubUrl" TEXT,
    "paperUrl" TEXT,
    "progressPercent" INTEGER,
    "stage" "ResearchStage" NOT NULL DEFAULT 'EXPLORING',
    "startYear" SMALLINT,
    "startMonth" SMALLINT,
    "endYear" SMALLINT,
    "endMonth" SMALLINT,
    "isCurrent" BOOLEAN NOT NULL DEFAULT false,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "publishedAt" TIMESTAMPTZ(3),
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ResearchProject_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ResearchQuestion" (
    "id" UUID NOT NULL,
    "researchProjectId" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "question" TEXT NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ResearchQuestion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Publication" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "authors" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "venue" TEXT,
    "type" "PublicationType" NOT NULL,
    "year" SMALLINT,
    "abstract" TEXT,
    "doi" TEXT,
    "arxivUrl" TEXT,
    "paperUrl" TEXT,
    "citation" TEXT,
    "bibtex" TEXT,
    "stage" "PublicationStage" NOT NULL,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "publishedAt" TIMESTAMPTZ(3),
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Publication_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Project" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "educationId" UUID,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "shortTitle" TEXT,
    "type" TEXT,
    "category" TEXT,
    "summary" TEXT NOT NULL,
    "description" TEXT,
    "problemStatement" TEXT,
    "motivation" TEXT,
    "approach" TEXT,
    "architecture" TEXT,
    "aiModels" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "datasets" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "challenges" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "results" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "lessons" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "institution" TEXT,
    "advisor" TEXT,
    "role" TEXT,
    "implementation" TEXT,
    "repositoryUrl" TEXT,
    "demoUrl" TEXT,
    "documentationUrl" TEXT,
    "seoTitle" TEXT,
    "seoDescription" TEXT,
    "projectStatus" "ProjectStatus",
    "startYear" SMALLINT,
    "startMonth" SMALLINT,
    "endYear" SMALLINT,
    "endMonth" SMALLINT,
    "isCurrent" BOOLEAN NOT NULL DEFAULT false,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "publishedAt" TIMESTAMPTZ(3),
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Project_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProjectContribution" (
    "id" UUID NOT NULL,
    "projectId" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "label" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ProjectContribution_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProjectFeature" (
    "id" UUID NOT NULL,
    "projectId" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ProjectFeature_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProjectTechnology" (
    "id" UUID NOT NULL,
    "projectId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ProjectTechnology_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Experience" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "type" "ExperienceType" NOT NULL,
    "organization" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "location" TEXT,
    "summary" TEXT NOT NULL,
    "domain" TEXT,
    "practiceAreas" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "highlights" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "startYear" SMALLINT,
    "startMonth" SMALLINT,
    "endYear" SMALLINT,
    "endMonth" SMALLINT,
    "isCurrent" BOOLEAN NOT NULL DEFAULT false,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Experience_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Education" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "institution" TEXT NOT NULL,
    "degree" TEXT NOT NULL,
    "field" TEXT,
    "location" TEXT,
    "startYear" SMALLINT,
    "startMonth" SMALLINT,
    "endYear" SMALLINT,
    "endMonth" SMALLINT,
    "isCurrent" BOOLEAN NOT NULL DEFAULT false,
    "gpa" DECIMAL(4,2),
    "gpaScale" DECIMAL(4,2),
    "coursework" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "thesis" TEXT,
    "activities" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "url" TEXT,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Education_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillCategory" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "SkillCategory_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Skill" (
    "id" UUID NOT NULL,
    "categoryId" UUID NOT NULL,
    "seedKey" TEXT,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Skill_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Certification" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "issuer" TEXT NOT NULL,
    "credentialId" TEXT,
    "credentialUrl" TEXT,
    "description" TEXT,
    "issueYear" SMALLINT,
    "issueMonth" SMALLINT,
    "expiryYear" SMALLINT,
    "expiryMonth" SMALLINT,
    "doesNotExpire" BOOLEAN NOT NULL DEFAULT false,
    "fileAssetId" UUID,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Certification_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Article" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "excerpt" TEXT,
    "content" TEXT NOT NULL,
    "contentFormat" TEXT NOT NULL DEFAULT 'markdown',
    "category" TEXT,
    "estimatedReadingMinutes" INTEGER,
    "seoTitle" TEXT,
    "seoDescription" TEXT,
    "featured" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "publishedAt" TIMESTAMPTZ(3),
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Article_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Tag" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Tag_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MediaAsset" (
    "id" UUID NOT NULL,
    "storageKey" TEXT NOT NULL,
    "provider" TEXT,
    "publicUrl" TEXT,
    "originalFilename" TEXT NOT NULL,
    "mimeType" TEXT NOT NULL,
    "byteSize" BIGINT,
    "width" INTEGER,
    "height" INTEGER,
    "altText" TEXT,
    "caption" TEXT,
    "checksum" TEXT,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "MediaAsset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProjectMedia" (
    "id" UUID NOT NULL,
    "projectId" UUID NOT NULL,
    "mediaAssetId" UUID NOT NULL,
    "role" "MediaRole" NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "altOverride" TEXT,
    "caption" TEXT,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ProjectMedia_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ResearchMedia" (
    "id" UUID NOT NULL,
    "researchProjectId" UUID NOT NULL,
    "mediaAssetId" UUID NOT NULL,
    "role" "MediaRole" NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "altOverride" TEXT,
    "caption" TEXT,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ResearchMedia_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ArticleMedia" (
    "id" UUID NOT NULL,
    "articleId" UUID NOT NULL,
    "mediaAssetId" UUID NOT NULL,
    "role" "MediaRole" NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,
    "altOverride" TEXT,
    "caption" TEXT,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ArticleMedia_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Resume" (
    "id" UUID NOT NULL,
    "profileId" UUID NOT NULL,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "version" TEXT,
    "fileAssetId" UUID,
    "isCurrent" BOOLEAN NOT NULL DEFAULT false,
    "status" "ContentStatus" NOT NULL DEFAULT 'DRAFT',
    "source" "ContentSource" NOT NULL DEFAULT 'ADMIN',
    "publishedAt" TIMESTAMPTZ(3),
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "Resume_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ContactSubmission" (
    "id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "organization" TEXT,
    "subject" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "status" "ContactSubmissionStatus" NOT NULL DEFAULT 'NEW',
    "hashedIp" TEXT,
    "userAgent" TEXT,
    "receivedAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "readAt" TIMESTAMPTZ(3),
    "repliedAt" TIMESTAMPTZ(3),
    "archivedAt" TIMESTAMPTZ(3),
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "ContactSubmission_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SiteSetting" (
    "id" UUID NOT NULL,
    "key" TEXT NOT NULL,
    "value" JSONB NOT NULL,
    "description" TEXT,
    "createdAt" TIMESTAMPTZ(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(3) NOT NULL,

    CONSTRAINT "SiteSetting_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "_ResearchProjectInterests" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL,

    CONSTRAINT "_ResearchProjectInterests_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateTable
CREATE TABLE "_ResearchPublications" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL,

    CONSTRAINT "_ResearchPublications_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateTable
CREATE TABLE "_ProjectResearch" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL,

    CONSTRAINT "_ProjectResearch_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateTable
CREATE TABLE "_ExperienceSkills" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL,

    CONSTRAINT "_ExperienceSkills_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateTable
CREATE TABLE "_ArticleToTag" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL,

    CONSTRAINT "_ArticleToTag_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateIndex
CREATE UNIQUE INDEX "Profile_key_key" ON "Profile"("key");

-- CreateIndex
CREATE UNIQUE INDEX "Profile_slug_key" ON "Profile"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "Profile_avatarId_key" ON "Profile"("avatarId");

-- CreateIndex
CREATE INDEX "SocialLink_profileId_status_sortOrder_idx" ON "SocialLink"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "SocialLink_profileId_key_key" ON "SocialLink"("profileId", "key");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchInterest_seedKey_key" ON "ResearchInterest"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchInterest_slug_key" ON "ResearchInterest"("slug");

-- CreateIndex
CREATE INDEX "ResearchInterest_profileId_status_sortOrder_idx" ON "ResearchInterest"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchProject_seedKey_key" ON "ResearchProject"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchProject_slug_key" ON "ResearchProject"("slug");

-- CreateIndex
CREATE INDEX "ResearchProject_profileId_status_featured_sortOrder_idx" ON "ResearchProject"("profileId", "status", "featured", "sortOrder");

-- CreateIndex
CREATE INDEX "ResearchProject_stage_status_idx" ON "ResearchProject"("stage", "status");

-- CreateIndex
CREATE INDEX "ResearchQuestion_researchProjectId_sortOrder_idx" ON "ResearchQuestion"("researchProjectId", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchQuestion_researchProjectId_key_key" ON "ResearchQuestion"("researchProjectId", "key");

-- CreateIndex
CREATE UNIQUE INDEX "Publication_slug_key" ON "Publication"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "Publication_doi_key" ON "Publication"("doi");

-- CreateIndex
CREATE INDEX "Publication_profileId_status_featured_sortOrder_idx" ON "Publication"("profileId", "status", "featured", "sortOrder");

-- CreateIndex
CREATE INDEX "Publication_stage_year_idx" ON "Publication"("stage", "year");

-- CreateIndex
CREATE UNIQUE INDEX "Project_seedKey_key" ON "Project"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "Project_slug_key" ON "Project"("slug");

-- CreateIndex
CREATE INDEX "Project_profileId_status_featured_sortOrder_idx" ON "Project"("profileId", "status", "featured", "sortOrder");

-- CreateIndex
CREATE INDEX "Project_educationId_idx" ON "Project"("educationId");

-- CreateIndex
CREATE INDEX "ProjectContribution_projectId_sortOrder_idx" ON "ProjectContribution"("projectId", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "ProjectContribution_projectId_key_key" ON "ProjectContribution"("projectId", "key");

-- CreateIndex
CREATE INDEX "ProjectFeature_projectId_sortOrder_idx" ON "ProjectFeature"("projectId", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "ProjectFeature_projectId_key_key" ON "ProjectFeature"("projectId", "key");

-- CreateIndex
CREATE INDEX "ProjectTechnology_projectId_sortOrder_idx" ON "ProjectTechnology"("projectId", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "ProjectTechnology_projectId_slug_key" ON "ProjectTechnology"("projectId", "slug");

-- CreateIndex
CREATE UNIQUE INDEX "Experience_seedKey_key" ON "Experience"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "Experience_slug_key" ON "Experience"("slug");

-- CreateIndex
CREATE INDEX "Experience_profileId_status_sortOrder_idx" ON "Experience"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "Education_seedKey_key" ON "Education"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "Education_slug_key" ON "Education"("slug");

-- CreateIndex
CREATE INDEX "Education_profileId_status_sortOrder_idx" ON "Education"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "SkillCategory_seedKey_key" ON "SkillCategory"("seedKey");

-- CreateIndex
CREATE UNIQUE INDEX "SkillCategory_slug_key" ON "SkillCategory"("slug");

-- CreateIndex
CREATE INDEX "SkillCategory_profileId_status_sortOrder_idx" ON "SkillCategory"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "Skill_seedKey_key" ON "Skill"("seedKey");

-- CreateIndex
CREATE INDEX "Skill_categoryId_status_sortOrder_idx" ON "Skill"("categoryId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "Skill_categoryId_slug_key" ON "Skill"("categoryId", "slug");

-- CreateIndex
CREATE UNIQUE INDEX "Certification_slug_key" ON "Certification"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "Certification_fileAssetId_key" ON "Certification"("fileAssetId");

-- CreateIndex
CREATE INDEX "Certification_profileId_status_sortOrder_idx" ON "Certification"("profileId", "status", "sortOrder");

-- CreateIndex
CREATE UNIQUE INDEX "Article_slug_key" ON "Article"("slug");

-- CreateIndex
CREATE INDEX "Article_profileId_status_featured_sortOrder_idx" ON "Article"("profileId", "status", "featured", "sortOrder");

-- CreateIndex
CREATE INDEX "Article_publishedAt_idx" ON "Article"("publishedAt");

-- CreateIndex
CREATE INDEX "Tag_profileId_name_idx" ON "Tag"("profileId", "name");

-- CreateIndex
CREATE UNIQUE INDEX "Tag_profileId_slug_key" ON "Tag"("profileId", "slug");

-- CreateIndex
CREATE UNIQUE INDEX "MediaAsset_storageKey_key" ON "MediaAsset"("storageKey");

-- CreateIndex
CREATE INDEX "MediaAsset_checksum_idx" ON "MediaAsset"("checksum");

-- CreateIndex
CREATE INDEX "ProjectMedia_projectId_sortOrder_idx" ON "ProjectMedia"("projectId", "sortOrder");

-- CreateIndex
CREATE INDEX "ProjectMedia_mediaAssetId_idx" ON "ProjectMedia"("mediaAssetId");

-- CreateIndex
CREATE UNIQUE INDEX "ProjectMedia_projectId_mediaAssetId_role_key" ON "ProjectMedia"("projectId", "mediaAssetId", "role");

-- CreateIndex
CREATE INDEX "ResearchMedia_researchProjectId_sortOrder_idx" ON "ResearchMedia"("researchProjectId", "sortOrder");

-- CreateIndex
CREATE INDEX "ResearchMedia_mediaAssetId_idx" ON "ResearchMedia"("mediaAssetId");

-- CreateIndex
CREATE UNIQUE INDEX "ResearchMedia_researchProjectId_mediaAssetId_role_key" ON "ResearchMedia"("researchProjectId", "mediaAssetId", "role");

-- CreateIndex
CREATE INDEX "ArticleMedia_articleId_sortOrder_idx" ON "ArticleMedia"("articleId", "sortOrder");

-- CreateIndex
CREATE INDEX "ArticleMedia_mediaAssetId_idx" ON "ArticleMedia"("mediaAssetId");

-- CreateIndex
CREATE UNIQUE INDEX "ArticleMedia_articleId_mediaAssetId_role_key" ON "ArticleMedia"("articleId", "mediaAssetId", "role");

-- CreateIndex
CREATE UNIQUE INDEX "Resume_slug_key" ON "Resume"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "Resume_fileAssetId_key" ON "Resume"("fileAssetId");

-- CreateIndex
CREATE INDEX "Resume_profileId_status_isCurrent_idx" ON "Resume"("profileId", "status", "isCurrent");

-- CreateIndex
CREATE INDEX "ContactSubmission_status_receivedAt_idx" ON "ContactSubmission"("status", "receivedAt");

-- CreateIndex
CREATE INDEX "ContactSubmission_email_idx" ON "ContactSubmission"("email");

-- CreateIndex
CREATE UNIQUE INDEX "SiteSetting_key_key" ON "SiteSetting"("key");

-- CreateIndex
CREATE INDEX "_ResearchProjectInterests_B_index" ON "_ResearchProjectInterests"("B");

-- CreateIndex
CREATE INDEX "_ResearchPublications_B_index" ON "_ResearchPublications"("B");

-- CreateIndex
CREATE INDEX "_ProjectResearch_B_index" ON "_ProjectResearch"("B");

-- CreateIndex
CREATE INDEX "_ExperienceSkills_B_index" ON "_ExperienceSkills"("B");

-- CreateIndex
CREATE INDEX "_ArticleToTag_B_index" ON "_ArticleToTag"("B");

-- AddForeignKey
ALTER TABLE "Profile" ADD CONSTRAINT "Profile_avatarId_fkey" FOREIGN KEY ("avatarId") REFERENCES "MediaAsset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SocialLink" ADD CONSTRAINT "SocialLink_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ResearchInterest" ADD CONSTRAINT "ResearchInterest_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ResearchQuestion" ADD CONSTRAINT "ResearchQuestion_researchProjectId_fkey" FOREIGN KEY ("researchProjectId") REFERENCES "ResearchProject"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Publication" ADD CONSTRAINT "Publication_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Project" ADD CONSTRAINT "Project_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Project" ADD CONSTRAINT "Project_educationId_fkey" FOREIGN KEY ("educationId") REFERENCES "Education"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProjectContribution" ADD CONSTRAINT "ProjectContribution_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProjectFeature" ADD CONSTRAINT "ProjectFeature_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProjectTechnology" ADD CONSTRAINT "ProjectTechnology_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Experience" ADD CONSTRAINT "Experience_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Education" ADD CONSTRAINT "Education_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillCategory" ADD CONSTRAINT "SkillCategory_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Skill" ADD CONSTRAINT "Skill_categoryId_fkey" FOREIGN KEY ("categoryId") REFERENCES "SkillCategory"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_fileAssetId_fkey" FOREIGN KEY ("fileAssetId") REFERENCES "MediaAsset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Article" ADD CONSTRAINT "Article_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Tag" ADD CONSTRAINT "Tag_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProjectMedia" ADD CONSTRAINT "ProjectMedia_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProjectMedia" ADD CONSTRAINT "ProjectMedia_mediaAssetId_fkey" FOREIGN KEY ("mediaAssetId") REFERENCES "MediaAsset"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ResearchMedia" ADD CONSTRAINT "ResearchMedia_researchProjectId_fkey" FOREIGN KEY ("researchProjectId") REFERENCES "ResearchProject"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ResearchMedia" ADD CONSTRAINT "ResearchMedia_mediaAssetId_fkey" FOREIGN KEY ("mediaAssetId") REFERENCES "MediaAsset"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ArticleMedia" ADD CONSTRAINT "ArticleMedia_articleId_fkey" FOREIGN KEY ("articleId") REFERENCES "Article"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ArticleMedia" ADD CONSTRAINT "ArticleMedia_mediaAssetId_fkey" FOREIGN KEY ("mediaAssetId") REFERENCES "MediaAsset"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Resume" ADD CONSTRAINT "Resume_profileId_fkey" FOREIGN KEY ("profileId") REFERENCES "Profile"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Resume" ADD CONSTRAINT "Resume_fileAssetId_fkey" FOREIGN KEY ("fileAssetId") REFERENCES "MediaAsset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ResearchProjectInterests" ADD CONSTRAINT "_ResearchProjectInterests_A_fkey" FOREIGN KEY ("A") REFERENCES "ResearchInterest"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ResearchProjectInterests" ADD CONSTRAINT "_ResearchProjectInterests_B_fkey" FOREIGN KEY ("B") REFERENCES "ResearchProject"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ResearchPublications" ADD CONSTRAINT "_ResearchPublications_A_fkey" FOREIGN KEY ("A") REFERENCES "Publication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ResearchPublications" ADD CONSTRAINT "_ResearchPublications_B_fkey" FOREIGN KEY ("B") REFERENCES "ResearchProject"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ProjectResearch" ADD CONSTRAINT "_ProjectResearch_A_fkey" FOREIGN KEY ("A") REFERENCES "Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ProjectResearch" ADD CONSTRAINT "_ProjectResearch_B_fkey" FOREIGN KEY ("B") REFERENCES "ResearchProject"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ExperienceSkills" ADD CONSTRAINT "_ExperienceSkills_A_fkey" FOREIGN KEY ("A") REFERENCES "Experience"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ExperienceSkills" ADD CONSTRAINT "_ExperienceSkills_B_fkey" FOREIGN KEY ("B") REFERENCES "Skill"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ArticleToTag" ADD CONSTRAINT "_ArticleToTag_A_fkey" FOREIGN KEY ("A") REFERENCES "Article"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_ArticleToTag" ADD CONSTRAINT "_ArticleToTag_B_fkey" FOREIGN KEY ("B") REFERENCES "Tag"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Partial dates retain the month-level precision of the source records.
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_start_month_check" CHECK ("startMonth" IS NULL OR ("startYear" IS NOT NULL AND "startMonth" BETWEEN 1 AND 12));
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_end_month_check" CHECK ("endMonth" IS NULL OR ("endYear" IS NOT NULL AND "endMonth" BETWEEN 1 AND 12));
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_end_requires_start_check" CHECK ("endYear" IS NULL OR "startYear" IS NOT NULL);
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_date_order_check" CHECK ("endYear" IS NULL OR "startYear" IS NULL OR "endYear" > "startYear" OR ("endYear" = "startYear" AND ("endMonth" IS NULL OR "startMonth" IS NULL OR "endMonth" >= "startMonth")));
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_current_date_check" CHECK (NOT "isCurrent" OR ("endYear" IS NULL AND "endMonth" IS NULL));
ALTER TABLE "ResearchProject" ADD CONSTRAINT "ResearchProject_progress_check" CHECK ("progressPercent" IS NULL OR "progressPercent" BETWEEN 0 AND 100);

ALTER TABLE "Project" ADD CONSTRAINT "Project_start_month_check" CHECK ("startMonth" IS NULL OR ("startYear" IS NOT NULL AND "startMonth" BETWEEN 1 AND 12));
ALTER TABLE "Project" ADD CONSTRAINT "Project_end_month_check" CHECK ("endMonth" IS NULL OR ("endYear" IS NOT NULL AND "endMonth" BETWEEN 1 AND 12));
ALTER TABLE "Project" ADD CONSTRAINT "Project_end_requires_start_check" CHECK ("endYear" IS NULL OR "startYear" IS NOT NULL);
ALTER TABLE "Project" ADD CONSTRAINT "Project_date_order_check" CHECK ("endYear" IS NULL OR "startYear" IS NULL OR "endYear" > "startYear" OR ("endYear" = "startYear" AND ("endMonth" IS NULL OR "startMonth" IS NULL OR "endMonth" >= "startMonth")));
ALTER TABLE "Project" ADD CONSTRAINT "Project_current_date_check" CHECK (NOT "isCurrent" OR ("endYear" IS NULL AND "endMonth" IS NULL));

ALTER TABLE "Experience" ADD CONSTRAINT "Experience_start_month_check" CHECK ("startMonth" IS NULL OR ("startYear" IS NOT NULL AND "startMonth" BETWEEN 1 AND 12));
ALTER TABLE "Experience" ADD CONSTRAINT "Experience_end_month_check" CHECK ("endMonth" IS NULL OR ("endYear" IS NOT NULL AND "endMonth" BETWEEN 1 AND 12));
ALTER TABLE "Experience" ADD CONSTRAINT "Experience_end_requires_start_check" CHECK ("endYear" IS NULL OR "startYear" IS NOT NULL);
ALTER TABLE "Experience" ADD CONSTRAINT "Experience_date_order_check" CHECK ("endYear" IS NULL OR "startYear" IS NULL OR "endYear" > "startYear" OR ("endYear" = "startYear" AND ("endMonth" IS NULL OR "startMonth" IS NULL OR "endMonth" >= "startMonth")));
ALTER TABLE "Experience" ADD CONSTRAINT "Experience_current_date_check" CHECK (NOT "isCurrent" OR ("endYear" IS NULL AND "endMonth" IS NULL));

ALTER TABLE "Education" ADD CONSTRAINT "Education_start_month_check" CHECK ("startMonth" IS NULL OR ("startYear" IS NOT NULL AND "startMonth" BETWEEN 1 AND 12));
ALTER TABLE "Education" ADD CONSTRAINT "Education_end_month_check" CHECK ("endMonth" IS NULL OR ("endYear" IS NOT NULL AND "endMonth" BETWEEN 1 AND 12));
ALTER TABLE "Education" ADD CONSTRAINT "Education_end_requires_start_check" CHECK ("endYear" IS NULL OR "startYear" IS NOT NULL);
ALTER TABLE "Education" ADD CONSTRAINT "Education_date_order_check" CHECK ("endYear" IS NULL OR "startYear" IS NULL OR "endYear" > "startYear" OR ("endYear" = "startYear" AND ("endMonth" IS NULL OR "startMonth" IS NULL OR "endMonth" >= "startMonth")));
ALTER TABLE "Education" ADD CONSTRAINT "Education_current_date_check" CHECK (NOT "isCurrent" OR ("endYear" IS NULL AND "endMonth" IS NULL));
ALTER TABLE "Education" ADD CONSTRAINT "Education_gpa_check" CHECK (("gpa" IS NULL OR "gpa" >= 0) AND ("gpaScale" IS NULL OR "gpaScale" > 0) AND ("gpa" IS NULL OR "gpaScale" IS NULL OR "gpa" <= "gpaScale"));

ALTER TABLE "Certification" ADD CONSTRAINT "Certification_issue_month_check" CHECK ("issueMonth" IS NULL OR ("issueYear" IS NOT NULL AND "issueMonth" BETWEEN 1 AND 12));
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_expiry_month_check" CHECK ("expiryMonth" IS NULL OR ("expiryYear" IS NOT NULL AND "expiryMonth" BETWEEN 1 AND 12));
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_expiry_requires_issue_check" CHECK ("expiryYear" IS NULL OR "issueYear" IS NOT NULL);
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_date_order_check" CHECK ("expiryYear" IS NULL OR "issueYear" IS NULL OR "expiryYear" > "issueYear" OR ("expiryYear" = "issueYear" AND ("expiryMonth" IS NULL OR "issueMonth" IS NULL OR "expiryMonth" >= "issueMonth")));
ALTER TABLE "Certification" ADD CONSTRAINT "Certification_no_expiry_check" CHECK (NOT "doesNotExpire" OR ("expiryYear" IS NULL AND "expiryMonth" IS NULL));

-- Guard numeric metadata independently of future application validation.
ALTER TABLE "MediaAsset" ADD CONSTRAINT "MediaAsset_byte_size_check" CHECK ("byteSize" IS NULL OR "byteSize" >= 0);
ALTER TABLE "MediaAsset" ADD CONSTRAINT "MediaAsset_dimensions_check" CHECK (("width" IS NULL OR "width" > 0) AND ("height" IS NULL OR "height" > 0));
ALTER TABLE "Article" ADD CONSTRAINT "Article_reading_time_check" CHECK ("estimatedReadingMinutes" IS NULL OR "estimatedReadingMinutes" > 0);

-- A portfolio profile can expose at most one current resume at a time.
CREATE UNIQUE INDEX "Resume_profileId_current_key" ON "Resume"("profileId") WHERE "isCurrent" = true;
