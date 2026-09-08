import "dotenv/config";

import { PrismaPg } from "@prisma/adapter-pg";

import {
  ContentSource,
  ContentStatus,
  ExperienceType,
  PrismaClient,
  ResearchStage,
  SocialLinkKind,
} from "../src/generated/prisma/client";
import { portfolioSeed } from "./seed-data";

const directUrl = process.env.DIRECT_URL;

if (!directUrl) {
  throw new Error(
    "DIRECT_URL is required to seed PostgreSQL. Copy .env.example to .env and add a direct PostgreSQL connection string.",
  );
}

const adapter = new PrismaPg({
  connectionString: directUrl,
  connectionTimeoutMillis: 10_000,
  idleTimeoutMillis: 10_000,
  max: 1,
});
const prisma = new PrismaClient({ adapter, log: ["error"] });

const cvRecord = {
  source: ContentSource.ACADEMIC_CV,
  status: ContentStatus.PUBLISHED,
} as const;

const userProvidedRecord = {
  source: ContentSource.USER_PROVIDED,
  status: ContentStatus.PUBLISHED,
} as const;

async function seedPortfolio() {
  await prisma.$transaction(
    async (transaction) => {
      const bootstrapPublishedAt = new Date();
      const existingProfile = await transaction.profile.findUnique({
        where: { key: portfolioSeed.profile.key },
        select: { id: true, source: true },
      });
      const profile =
        existingProfile?.source === ContentSource.ADMIN
          ? existingProfile
          : await transaction.profile.upsert({
              where: { key: portfolioSeed.profile.key },
              create: {
                ...portfolioSeed.profile,
                positioning: [...portfolioSeed.profile.positioning],
                about: [...portfolioSeed.profile.about],
                ...userProvidedRecord,
              },
              update: {
                ...portfolioSeed.profile,
                positioning: [...portfolioSeed.profile.positioning],
                about: [...portfolioSeed.profile.about],
                ...userProvidedRecord,
              },
            });

      for (const link of portfolioSeed.socialLinks) {
        const existingLink = await transaction.socialLink.findUnique({
          where: {
            profileId_key: { profileId: profile.id, key: link.key },
          },
          select: { source: true },
        });
        if (existingLink?.source === ContentSource.ADMIN) continue;

        await transaction.socialLink.upsert({
          where: {
            profileId_key: { profileId: profile.id, key: link.key },
          },
          create: {
            ...link,
            kind: SocialLinkKind[link.kind],
            profileId: profile.id,
            ...cvRecord,
          },
          update: {
            ...link,
            kind: SocialLinkKind[link.kind],
            ...cvRecord,
          },
        });
      }

      const interests = [];
      for (const interest of portfolioSeed.researchInterests) {
        const existingInterest = await transaction.researchInterest.findUnique({
          where: { seedKey: interest.seedKey },
          select: { id: true, source: true },
        });
        const record =
          existingInterest?.source === ContentSource.ADMIN
            ? existingInterest
            : await transaction.researchInterest.upsert({
                where: { seedKey: interest.seedKey },
                create: { ...interest, profileId: profile.id, ...cvRecord },
                update: { ...interest, profileId: profile.id, ...cvRecord },
              });
        interests.push(record);
      }

      const { questions, ...researchData } = portfolioSeed.researchProject;
      const existingResearch = await transaction.researchProject.findUnique({
        where: { seedKey: researchData.seedKey },
        select: { id: true, publishedAt: true, source: true },
      });
      const reconcileResearch =
        existingResearch?.source !== ContentSource.ADMIN;
      const researchProject = reconcileResearch
        ? await transaction.researchProject.upsert({
            where: { seedKey: researchData.seedKey },
            create: {
              ...researchData,
              profileId: profile.id,
              stage: ResearchStage.EARLY_STAGE,
              isCurrent: true,
              featured: true,
              publishedAt: bootstrapPublishedAt,
              interests: { connect: interests.map(({ id }) => ({ id })) },
              ...cvRecord,
            },
            update: {
              ...researchData,
              profileId: profile.id,
              stage: ResearchStage.EARLY_STAGE,
              isCurrent: true,
              featured: true,
              publishedAt:
                existingResearch?.publishedAt ?? bootstrapPublishedAt,
              interests: { connect: interests.map(({ id }) => ({ id })) },
              ...cvRecord,
            },
          })
        : existingResearch;

      for (const question of reconcileResearch ? questions : []) {
        await transaction.researchQuestion.upsert({
          where: {
            researchProjectId_key: {
              researchProjectId: researchProject.id,
              key: question.key,
            },
          },
          create: { ...question, researchProjectId: researchProject.id },
          update: question,
        });
      }

      const { contributions, features, technologies, ...projectData } =
        portfolioSeed.project;
      const existingProject = await transaction.project.findUnique({
        where: { seedKey: projectData.seedKey },
        select: { id: true, publishedAt: true, source: true },
      });
      const reconcileProject = existingProject?.source !== ContentSource.ADMIN;
      const project = reconcileProject
        ? await transaction.project.upsert({
            where: { seedKey: projectData.seedKey },
            create: {
              ...projectData,
              profileId: profile.id,
              featured: true,
              publishedAt: bootstrapPublishedAt,
              ...cvRecord,
            },
            update: {
              ...projectData,
              profileId: profile.id,
              featured: true,
              publishedAt: existingProject?.publishedAt ?? bootstrapPublishedAt,
              ...cvRecord,
            },
          })
        : existingProject;

      for (const contribution of reconcileProject ? contributions : []) {
        await transaction.projectContribution.upsert({
          where: {
            projectId_key: { projectId: project.id, key: contribution.key },
          },
          create: { ...contribution, projectId: project.id },
          update: contribution,
        });
      }

      for (const feature of reconcileProject ? features : []) {
        await transaction.projectFeature.upsert({
          where: {
            projectId_key: { projectId: project.id, key: feature.key },
          },
          create: { ...feature, projectId: project.id },
          update: feature,
        });
      }

      for (const technology of reconcileProject ? technologies : []) {
        await transaction.projectTechnology.upsert({
          where: {
            projectId_slug: { projectId: project.id, slug: technology.slug },
          },
          create: { ...technology, projectId: project.id },
          update: technology,
        });
      }

      for (const experience of portfolioSeed.experiences) {
        const existingExperience = await transaction.experience.findUnique({
          where: { seedKey: experience.seedKey },
          select: { source: true },
        });
        if (existingExperience?.source === ContentSource.ADMIN) continue;

        await transaction.experience.upsert({
          where: { seedKey: experience.seedKey },
          create: {
            ...experience,
            practiceAreas: [...experience.practiceAreas],
            profileId: profile.id,
            type: ExperienceType.CLIENT_ENGAGEMENT,
            ...cvRecord,
          },
          update: {
            ...experience,
            practiceAreas: [...experience.practiceAreas],
            profileId: profile.id,
            type: ExperienceType.CLIENT_ENGAGEMENT,
            ...cvRecord,
          },
        });
      }

      for (const education of portfolioSeed.education) {
        const existingEducation = await transaction.education.findUnique({
          where: { seedKey: education.seedKey },
          select: { source: true },
        });
        if (existingEducation?.source === ContentSource.ADMIN) continue;

        await transaction.education.upsert({
          where: { seedKey: education.seedKey },
          create: {
            ...education,
            profileId: profile.id,
            isCurrent: false,
            ...cvRecord,
          },
          update: {
            ...education,
            profileId: profile.id,
            isCurrent: false,
            ...cvRecord,
          },
        });
      }

      for (const categoryData of portfolioSeed.skillCategories) {
        const { skills, ...category } = categoryData;
        const existingCategory = await transaction.skillCategory.findUnique({
          where: { seedKey: category.seedKey },
          select: { id: true, source: true },
        });
        const reconcileCategory =
          existingCategory?.source !== ContentSource.ADMIN;
        const skillCategory = reconcileCategory
          ? await transaction.skillCategory.upsert({
              where: { seedKey: category.seedKey },
              create: { ...category, profileId: profile.id, ...cvRecord },
              update: { ...category, profileId: profile.id, ...cvRecord },
            })
          : existingCategory;

        for (const [sortOrder, skill] of (
          reconcileCategory ? skills : []
        ).entries()) {
          const existingSkill = await transaction.skill.findUnique({
            where: { seedKey: skill.seedKey },
            select: { source: true },
          });
          if (existingSkill?.source === ContentSource.ADMIN) continue;

          await transaction.skill.upsert({
            where: { seedKey: skill.seedKey },
            create: {
              ...skill,
              categoryId: skillCategory.id,
              sortOrder,
              ...cvRecord,
            },
            update: {
              ...skill,
              categoryId: skillCategory.id,
              sortOrder,
              ...cvRecord,
            },
          });
        }
      }
    },
    { maxWait: 10_000, timeout: 60_000 },
  );

  console.info(
    "Portfolio seed complete: 1 profile, 2 social links, 4 research interests, 1 research direction, 5 research questions, 1 project, 2 client engagements, 2 education records, 7 skill categories, and 40 skills.",
  );
}

try {
  await seedPortfolio();
} catch {
  console.error(
    "Portfolio seed failed. Verify the direct database connection and migration status.",
  );
  process.exitCode = 1;
} finally {
  await prisma.$disconnect();
}
