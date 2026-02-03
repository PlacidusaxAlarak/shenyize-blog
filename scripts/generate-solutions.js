// scripts/generate-solutions.js
import fs from 'node:fs';
import path from 'node:path';

// --- 配置项 ---
// 确保路径正确指向你的 posts 目录
const POSTS_DIR = path.join(process.cwd(), 'src/content/posts');
const OUTPUT_FILE = path.join(POSTS_DIR, 'solutions.md');

// 1. 修正目录名称：必须与实际文件夹名称完全一致（Linux下区分大小写）
// 根据你的文件列表，CodeForces 的 F 是大写的
const TARGET_DIRS = ['Atcoder', 'CodeForces'];

// --- 核心：生成的 MD 文件模版 ---
function buildIndexFrontmatter(publishedDate) {
  return `---
title: "算法题解索引"
published: ${publishedDate}
description: "汇总所有 AtCoder 和 Codeforces 的算法题解索引。"
tags: ["算法", "Atcoder", "CodeForces"]
category: "算法"
priority: 9
draft: false
---

## 简介

这里收录了我在 **Codeforces** 和 **AtCoder** 刷题过程中积累的题解与心得。
此页面由脚本自动生成，实时更新。

:::note
点击下方的题目链接可直接跳转到对应文章。
:::

`;
}

// --- 辅助函数 ---
function extractFrontmatterBlock(content) {
  const match = content.match(/^---\s*([\s\S]+?)\s*---/);
  return match ? match[1] : "";
}

function normalizeFrontmatterValue(value) {
  let normalized = value.trim();
  if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith("'") && normalized.endsWith("'"))
  ) {
    normalized = normalized.slice(1, -1);
  }
  return normalized.replace(/\s+/g, " ").trim();
}

function parseFrontmatter(content) {
  const block = extractFrontmatterBlock(content);
  const lines = block.split("\n");
  const frontmatter = {};
  let currentKey = "";
  let currentValue = "";
  let quoteChar = "";

  for (const rawLine of lines) {
    const line = rawLine ?? "";
    if (!currentKey) {
      const colonIndex = line.indexOf(":");
      if (colonIndex === -1) continue;
      const key = line.slice(0, colonIndex).trim();
      let value = line.slice(colonIndex + 1).trim();
      if (!value) {
        frontmatter[key] = "";
        continue;
      }
      if (
        (value.startsWith('"') && !value.endsWith('"')) ||
        (value.startsWith("'") && !value.endsWith("'"))
      ) {
        currentKey = key;
        currentValue = value;
        quoteChar = value[0];
        continue;
      }
      frontmatter[key] = normalizeFrontmatterValue(value);
      continue;
    }

    currentValue += `\n${line}`;
    if (line.trimEnd().endsWith(quoteChar)) {
      frontmatter[currentKey] = normalizeFrontmatterValue(currentValue);
      currentKey = "";
      currentValue = "";
      quoteChar = "";
    }
  }

  if (currentKey) {
    frontmatter[currentKey] = normalizeFrontmatterValue(currentValue);
  }

  return frontmatter;
}

function normalizeDate(value) {
  const match = value.match(/\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : "";
}

function getFiles(dir) {
  const fullPath = path.join(POSTS_DIR, dir);
  if (!fs.existsSync(fullPath)) return [];

  const files = fs.readdirSync(fullPath).filter(file => file.endsWith('.md'));
  
  return files.map(file => {
    const content = fs.readFileSync(path.join(fullPath, file), 'utf-8');
    const fm = parseFrontmatter(content);
    
    // 获取文件名作为 slug (移除 .md)
    const slug = file.replace(/\.md$/, '');
    
    return {
      title: fm.title || slug,
      date: normalizeDate(fm.published) || '1970-01-01',
      // 2. 关键修改：生成链接时强制转为小写，以匹配 Astro 的路由规则
      link: `/posts/${dir.toLowerCase()}/${slug.toLowerCase()}/`, 
      filename: file
    };
  });
}

function getExistingPublishedDate() {
  if (!fs.existsSync(OUTPUT_FILE)) return "";
  const content = fs.readFileSync(OUTPUT_FILE, "utf-8");
  const fm = parseFrontmatter(content);
  return normalizeDate(fm.published);
}

// --- 主逻辑 ---
async function generate() {
  let markdownContent = "";
  const existingPublished = getExistingPublishedDate();
  let latestDate = "";

  for (const dir of TARGET_DIRS) {
    const posts = getFiles(dir);
    
    if (posts.length === 0) continue;

    posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    if (!latestDate) {
      latestDate = posts[0].date;
    } else if (new Date(posts[0].date).getTime() > new Date(latestDate).getTime()) {
      latestDate = posts[0].date;
    }

    markdownContent += `\n## ${dir} (${posts.length})\n\n`;
    
    posts.forEach(post => {
      markdownContent += `- [${post.title}](${post.link}) <small style="color:gray">${post.date}</small>\n`;
    });
  }

  const publishedDate =
    existingPublished ||
    latestDate ||
    new Date().toISOString().split("T")[0];
  const indexFrontmatter = buildIndexFrontmatter(publishedDate);
  const finalContent = `${indexFrontmatter}${markdownContent}`;

  fs.writeFileSync(OUTPUT_FILE, finalContent, 'utf-8');
  console.log(`✅ 题解索引已更新 (Priority: 9): ${OUTPUT_FILE}`);
}

generate();
