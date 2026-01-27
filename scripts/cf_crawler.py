import requests
from bs4 import BeautifulSoup
import html2text
import os
from datetime import datetime

def crawl_codeforces(url):
    """
    爬取 CodeForces 题目并生成博文 Markdown 文件
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 提取题目主体
        problem_body = soup.find('div', class_='problem-statement')
        if not problem_body:
            print("未能找到题目内容，请检查链接是否正确。")
            return

        # 提取标题和 ID
        title_tag = problem_body.find('div', class_='title')
        title = title_tag.text.strip() if title_tag else "Unknown Title"
        
        # 从 URL 提取 ID，例如 https://codeforces.com/contest/1927/problem/A -> 1927A
        parts = url.split('/')
        problem_id = parts[-3] + parts[-1] if 'contest' in url else parts[-1]
        
        # 转换 HTML 为 Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0  # 不限制行宽
        
        # 移除 header 避免重复信息
        header = problem_body.find('div', class_='header')
        content_html = str(problem_body).replace(str(header), "") if header else str(problem_body)
        markdown_content = h.handle(content_html)

        # 构造符合博客 schema 的 Frontmatter
        frontmatter = f"""---
title: "CodeForces {problem_id}: {title}"
published: {datetime.now().strftime('%Y-%m-%d')}
description: "CodeForces 算法题解：{title}"
tags: ["CodeForces", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[{url}]({url})

## 题目大意
## 解题思路
## 代码实现
```cpp
#include <iostream>
using namespace std;

int main() {{
    // 在此输入你的代码
    return 0;
}}
```
题目原文
{markdown_content} """

        # 确保目录存在
        save_dir = "src/content/posts/CodeForces"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f"{problem_id}.md"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(frontmatter)
        print(f"成功生成题解模板: {filepath}")

    except Exception as e:
        print(f"爬取失败: {e}")
    
if __name__ == "__main__": 
    url = input("请输入 CodeForces 题目链接 (如 https://codeforces.com/problemset/problem/1927/A): ") 
    crawl_codeforces(url)