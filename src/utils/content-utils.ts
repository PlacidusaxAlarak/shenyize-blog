// src/utils/content-utils.ts
import { type CollectionEntry, getCollection } from "astro:content";
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import { getCategoryUrl } from "@utils/url-utils.ts";

// Retrieve posts and sort them by priority (primary) and publication date (secondary)
async function getRawSortedPosts() {
    const allBlogPosts = await getCollection("posts", ({ data }) => {
        return import.meta.env.PROD ? data.draft !== true : true;
    });

    const sorted = allBlogPosts.sort((a, b) => {
        // 获取优先级，如果未设置则默认为 0
        const priorityA = a.data.priority ?? 0;
        const priorityB = b.data.priority ?? 0;

        // 1. 首先按照优先级排序 (10 最高，0 最低)
        if (priorityA !== priorityB) {
            return priorityB - priorityA;
        }

        // 2. 如果优先级相同，则按照发布日期降序排序
        const dateA = new Date(a.data.published);
        const dateB = new Date(b.data.published);
        return dateB.getTime() - dateA.getTime();
    });
    return sorted;
}

export async function getSortedPosts() {
    const sorted = await getRawSortedPosts();

    for (let i = 1; i < sorted.length; i++) {
        sorted[i].data.nextSlug = sorted[i - 1].slug;
        sorted[i].data.nextTitle = sorted[i - 1].data.title;
    }
    for (let i = 0; i < sorted.length - 1; i++) {
        sorted[i].data.prevSlug = sorted[i + 1].slug;
        sorted[i].data.prevTitle = sorted[i + 1].data.title;
    }

    return sorted;
}

export type PostForList = {
    slug: string;
    data: CollectionEntry<"posts">["data"];
};

export async function getSortedPostsList(): Promise<PostForList[]> {
    const sortedFullPosts = await getRawSortedPosts();

    // delete post.body
    const sortedPostsList = sortedFullPosts.map((post) => ({
        slug: post.slug,
        data: post.data,
    }));

    return sortedPostsList;
}

export type Tag = {
    name: string;
    count: number;
};

export async function getTagList(): Promise<Tag[]> {
    const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
        return import.meta.env.PROD ? data.draft !== true : true;
    });

    const countMap: { [key: string]: number } = {};
    allBlogPosts.forEach((post: { data: { tags: string[] } }) => {
        post.data.tags.forEach((tag: string) => {
            if (!countMap[tag]) countMap[tag] = 0;
            countMap[tag]++;
        });
    });

    // sort tags
    const keys: string[] = Object.keys(countMap).sort((a, b) => {
        return a.toLowerCase().localeCompare(b.toLowerCase());
    });

    return keys.map((key) => ({ name: key, count: countMap[key] }));
}

export type Category = {
    name: string;
    count: number;
    url: string;
};

export async function getCategoryList(): Promise<Category[]> {
    const allBlogPosts = await getCollection<"posts">("posts", ({ data }) => {
        return import.meta.env.PROD ? data.draft !== true : true;
    });
    const count: { [key: string]: number } = {};
    allBlogPosts.forEach((post: { data: { category: string | null } }) => {
        if (!post.data.category) {
            const ucKey = i18n(I18nKey.uncategorized);
            count[ucKey] = count[ucKey] ? count[ucKey] + 1 : 1;
            return;
        }

        const categoryName =
            typeof post.data.category === "string"
                ? post.data.category.trim()
                : String(post.data.category).trim();

        count[categoryName] = count[categoryName] ? count[categoryName] + 1 : 1;
    });

    const lst = Object.keys(count).sort((a, b) => {
        return a.toLowerCase().localeCompare(b.toLowerCase());
    });

    const ret: Category[] = [];
    for (const c of lst) {
        ret.push({
            name: c,
            count: count[c],
            url: getCategoryUrl(c),
        });
    }
    return ret;
}

// --- 请添加到 src/utils/content-utils.ts 文件的末尾 ---

export async function getSortedSolutions() {
    // 获取所有文章
    const sorted = await getRawSortedPosts();
    
    // 过滤逻辑：只保留分类为 "算法" 的文章，或者是包含 "AtCoder"/"Codeforces" 标签的文章
    // 你可以根据需要修改这里的判断条件
    const solutions = sorted.filter(post => 
        post.data.category === '算法' || 
        post.data.tags?.includes('AtCoder') || 
        post.data.tags?.includes('Codeforces')
    );

    // 为筛选出来的题解列表重新计算 "上一篇/下一篇"
    // 这样在浏览题解时，点击 "下一篇" 会跳转到下一个题解，而不是跳转到其他生活类博客
    for (let i = 1; i < solutions.length; i++) {
        solutions[i].data.nextSlug = solutions[i - 1].slug;
        solutions[i].data.nextTitle = solutions[i - 1].data.title;
    }
    for (let i = 0; i < solutions.length - 1; i++) {
        solutions[i].data.prevSlug = solutions[i + 1].slug;
        solutions[i].data.prevTitle = solutions[i + 1].data.title;
    }

    return solutions;
}