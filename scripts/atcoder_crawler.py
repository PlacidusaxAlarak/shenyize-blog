import requests
from bs4 import BeautifulSoup
import html2text
import os
from datetime import datetime

def crawl_atcoder(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # 提取标题 (例如: A - Welcome to AtCoder)
    title = soup.find('span', class_='h2').text.strip()
    problem_id = url.split('/')[-1].upper()
    
    # AtCoder 题目内容在 #task-statement 中，通常有分语言的 span
    task_statement = soup.find('div', id='task-statement')
    # 尝试寻找英文版，如果找不到则取全部
    content_section = task_statement.find('span', class_='lang-en')
    if not content_section:
        content_section = task_statement

    h = html2text.HTML2Text()
    markdown_content = h.handle(str(content_section))

    # 准备 Frontmatter
    # 注意：f-string 内部的每一行都必须顶格写，否则 Markdown 会识别出错
    frontmatter = f"""---
title: "AtCoder {problem_id}: {title}"
published: {datetime.now().strftime('%Y-%m-%d')}
description: "AtCoder 算法题解：{title}"
tags: ["AtCoder", "算法"]
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
"""

    # 保存文件
    save_dir = "src/content/posts/Atcoder"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{problem_id}.md"
    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
        f.write(frontmatter)
    print(f"成功生成: {filename}")
if __name__ == "__main__": 
    url = input("请输入 AtCoder 题目链接: ") 
    crawl_atcoder(url)