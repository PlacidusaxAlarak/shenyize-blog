---
title: "AtCoder ABC450_D: D - Minimize Range
			Editorial"
published: 2026-03-25
description: "AtCoder 算法题解：D - Minimize Range
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc450/tasks/abc450_d](https://atcoder.jp/contests/abc450/tasks/abc450_d)

## 题目大意
给定一个长度为 `N` 的正整数序列 `A` 和一个正整数 `K`。

你可以进行任意多次操作，每次选择一个下标 `i`，将 `A_i` 加上 `K`。

要求求出最终能够得到的最小值：
`max(A) - min(A)`。

## 解题思路
这道题的关键在于：无论对某个数加多少次 `K`，它对 `K` 取模后的余数都不会改变。

因此，一个数真正“固定”的只有它的余数，至于它具体落在这个余数对应的哪一层上，是可以通过不断加 `K` 来调整的。

我们先把所有数都变成 `a[i] % k`，然后排序。

设排序后的余数序列为：
`b_1 <= b_2 <= ... <= b_n`

如果把这些余数看成圆环 `mod K` 上的点，那么我们最终其实是在这个圆环上选择一个断点，把一部分数保留在当前这一圈，另一部分数加上一个 `K` 放到下一圈。

这样做之后，所有数都会落在一个长度小于等于 `K` 的连续区间中，我们要做的就是让这个区间尽可能短。

换个角度看：

- 如果我们保留了圆环上某一段“最大的空隙”不用，
- 那么剩下所有点所覆盖的最短区间长度，
- 就等于 `K - 最大空隙长度`。

所以问题就转化成了：

1. 先求出排序后相邻余数之间的差值 `b_{i+1} - b_i`；
2. 还要额外考虑首尾环绕形成的间隔 `b_1 + K - b_n`；
3. 找出这些间隔中的最大值 `max_gap`；
4. 答案就是 `K - max_gap`。

代码中的：

- `a[i] %= k` 表示只保留每个数的余数；
- `sort(a+1, a+n+1)` 对所有余数排序；
- `a[1] + k - a[n]` 处理首尾环绕的间隔；
- `a[i+1] - a[i]` 处理相邻余数之间的间隔；
- 最后输出 `k - maxx`，也就是最短可能区间长度。

时间复杂度为 `O(N log N)`，瓶颈在排序；空间复杂度为 `O(N)`。

## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=2e5+10;
int n;
ll k;
ll a[N];
int main(){
    scanf("%d%lld", &n, &k);
    for(int i=1;i<=n;i++) {
        scanf("%lld", &a[i]);
        a[i]%=k;
    }
    sort(a+1, a+n+1);
    ll maxx=a[1]+k-a[n];//首尾环绕
    for(int i=1;i<n;i++) {
        maxx=max(maxx, a[i+1]-a[i]);
    }
    printf("%lld\n", k-maxx);
    return 0;
}
```
