import { defineCollection, z } from 'astro:content';

const postsCollection = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        published: z.date(),
        updated: z.date().optional(),
        draft: z.boolean().optional(),
        description: z.string().optional(),
        image: z.string().optional(),
        tags: z.array(z.string()).optional(),
        category: z.string().optional(),
        lang: z.string().optional(),
        /* 新增：优先级字段，默认为 0，范围 0-10 */
        priority: z.number().min(0).max(10).optional().default(0),
        /* 保留你之前的导航字段 */
        prevTitle: z.string().default(""),
        prevSlug: z.string().default(""),
        nextTitle: z.string().default(""),
        nextSlug: z.string().default(""),
    }),
});

const specCollection = defineCollection({
    schema: z.object({}),
});

export const collections = {
    posts: postsCollection,
    spec: specCollection,
};