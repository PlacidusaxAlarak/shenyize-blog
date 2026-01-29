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
const INDEX_FRONTMATTER = `---
title: "算法题解索引"
published: ${new Date().toISOString().split('T')[0]}
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

// --- 辅助函数 ---
function parseFrontmatter(content) {
  const match = content.match(/^---\s+([\s\S]+?)\s+---/);
  if (!match) return {};
  
  const frontmatter = {};
  const lines = match[1].split('\n');
  for (const line of lines) {
    const colonIndex = line.indexOf(':');
    if (colonIndex !== -1) {
      const key = line.slice(0, colonIndex).trim();
      let value = line.slice(colonIndex + 1).trim();
      if (value.startsWith('"') && value.endsWith('"')) {
        value = value.slice(1, -1);
      }
      frontmatter[key] = value;
    }
  }
  return frontmatter;
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
      date: fm.published || '1970-01-01',
      // 2. 关键修改：生成链接时强制转为小写，以匹配 Astro 的路由规则
      link: `/posts/${dir.toLowerCase()}/${slug.toLowerCase()}/`, 
      filename: file
    };
  });
}

// --- 主逻辑 ---
async function generate() {
  let markdownContent = INDEX_FRONTMATTER;

  for (const dir of TARGET_DIRS) {
    const posts = getFiles(dir);
    
    if (posts.length === 0) continue;

    posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    markdownContent += `\n## ${dir} (${posts.length})\n\n`;
    
    posts.forEach(post => {
      markdownContent += `- [${post.title}](${post.link}) <small style="color:gray">${post.date}</small>\n`;
    });
  }

  fs.writeFileSync(OUTPUT_FILE, markdownContent, 'utf-8');
  console.log(`✅ 题解索引已更新 (Priority: 9): ${OUTPUT_FILE}`);
}

generate();