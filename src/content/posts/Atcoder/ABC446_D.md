---
title: "AtCoder ABC446_D: D - Max Straight
			Editorial"
published: 2026-03-04
description: "AtCoder 算法题解：D - Max Straight
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc446/tasks/abc446_d](https://atcoder.jp/contests/abc446/tasks/abc446_d)

## 题目大意


给定一个长度为 $N$ 的整数序列 $A$。

你需要从序列 $A$ 中找出一个子序列（不要求元素在原序列中连续，但需要保持原有的先后顺序），使得这个子序列是一个**公差为 1 的递增等差数列**（即形如 $x, x+1, x+2, \dots$ 的形式）。

求满足上述条件的**最长**子序列的长度。
## 解题思路
这是一个简单的 DP 转移问题。设 `dp[x]` 表示**以值 `x` 结尾**的、满足“相邻元素差为 1 且递增”的最长子序列长度。
遍历数组时，当前值 `a[i]` 只能从 `a[i]-1` 转移，因此有：
`dp[a[i]] = dp[a[i]-1] + 1`，若 `a[i]-1` 不存在则视为 0，从而长度为 1。
由于值域可达 `1e9`，用 `unordered_map` 维护 `dp`，并在遍历中更新答案即可。
## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+10;
int n, maxx=-0x3f3f3f3f;
int a[N];
unordered_map<int, int> q;
int main(){
    scanf("%d", &n);
    for(int i=1;i<=n;i++) scanf("%d", &a[i]);
    for(int i=1;i<=n;i++){
        if(q[a[i]-1]==0) q[a[i]]=1;
        else q[a[i]]=q[a[i]-1]+1;
    }
    for(auto u:q){
        maxx=max(maxx, u.second);
    }
    printf("%d\n", maxx);
    return 0;
}
```
