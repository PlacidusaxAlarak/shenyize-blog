// scripts/generate-solutions.js
import fs from 'node:fs';
import path from 'node:path';

// --- 配置项 ---
// 确保路径正确指向你的 posts 目录
const POSTS_DIR = path.join(process.cwd(), 'src/content/posts');
const OUTPUT_FILE = path.join(POSTS_DIR, 'solutions.md');

// 需要扫描的子目录名称
const TARGET_DIRS = ['Atcoder', 'Codeforces'];

// --- 核心：生成的 MD 文件模版 ---
// 这里设置了 priority: 9，以及简单的介绍文案
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

// 简单的 Frontmatter 解析器，提取 title 和 published
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
      // 去除引号
      if (value.startsWith('"') && value.endsWith('"')) {
        value = value.slice(1, -1);
      }
      frontmatter[key] = value;
    }
  }
  return frontmatter;
}

// 扫描目录获取文章列表
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
      // 构建文章链接：/posts/目录名/文件名
      link: `/posts/${dir}/${slug}/`, 
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

    // 按日期倒序排列（最新的在前面）
    posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    markdownContent += `\n## ${dir} (${posts.length})\n\n`;
    
    posts.forEach(post => {
      // 生成 Markdown 列表项
      markdownContent += `- [${post.title}](${post.link}) <small style="color:gray">${post.date}</small>\n`;
    });
  }

  // 写入文件
  fs.writeFileSync(OUTPUT_FILE, markdownContent, 'utf-8');
  console.log(`✅ 题解索引已更新 (Priority: 9): ${OUTPUT_FILE}`);
}

generate();