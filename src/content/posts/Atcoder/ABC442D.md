---
title: "AtCoder ABC442_D: D - Swap and Range Sum
			Editorial"
published: 2026-01-27
description: "AtCoder 算法题解：D - Swap and Range Sum
			Editorial"
tags: ["Atcoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc442/tasks/abc442_d](https://atcoder.jp/contests/abc442/tasks/abc442_d)

## 题目大意
给定一个长度为 $N$ 的整数序列 $A = (A_1, A_2, \dots, A_N)$。

你需要按顺序处理 $Q$ 个查询，查询分为以下两种类型：

- **修改操作** `1 x`：交换序列中第 $x$ 个元素 $A_x$ 和第 $x+1$ 个元素 $A_{x+1}$ 的值（其中 $1 \le x < N$）。
- **查询操作** `2 l r`：计算并输出区间 $[l, r]$ 内所有元素的和，即 $\sum_{i=l}^{r} A_i$。

**数据范围与约束：**
- $2 \le N \le 2 \times 10^5$
- $1 \le Q \le 5 \times 10^5$


## 解题思路
对于此题，我们需要考虑交换 $x$ 与 $x+1$ 会对前缀和造成什么后果。

令 $sum$ 为前缀和数组。

- 对于 $1 \le i \le x-1$，$sum_i$ 还是和原来未交换的结果一致，为 $\sum_{j=1}^{i} A_j$。
- 对于 $i \ge x+1$，$sum_i$ 也和原来未交换的结果一致（因为集合元素没变，只是顺序变了），为 $\sum_{j=1}^{i} A_j$。

**受到影响的只有 $sum_x$：**

- 原 $sum_x = \sum_{j=1}^{x} A_j$
- 交换后的 $sum_x = \sum_{j=1}^{x-1} A_j + A_{x+1}$

**结论：**

1.  当 `operator == 1` 时，我们只需要更新 $sum_x$，并记得在原数组中交换 $A_x$ 与 $A_{x+1}$。
2.  当 `operator == 2` 时，正常利用前缀和计算区间和即可。
## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+10;
typedef long long ll;
ll sum[N], a[N];
int n, q, op, l, r, x;
int main(){
	scanf("%d%d", &n, &q);
	for(int i=1;i<=n;i++) {
		scanf("%lld", &a[i]);
		sum[i]=sum[i-1]+a[i];
	}
	while(q--){
		scanf("%d", &op);
		if(op==1) {
			scanf("%d", &x);
			sum[x]=sum[x-1]+a[x+1];
			swap(a[x], a[x+1]);
		}
		else {
			scanf("%d%d", &l, &r);
			printf("%lld\n", sum[r]-sum[l-1]);
		}
	}
	return 0; 
} 
```
