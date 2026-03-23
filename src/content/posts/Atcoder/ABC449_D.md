---
title: "AtCoder ABC449_D: D - Make Target 2 Editorial"
published: 2026-03-23
description: "AtCoder 算法题解：D - Make Target 2 Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc449/tasks/abc449_d](https://atcoder.jp/contests/abc449/tasks/abc449_d)

## 题目大意
在二维坐标平面上，若整点 `(x, y)` 满足 `max(|x|, |y|)` 为偶数，则该点会被染成黑色；否则会被染成白色。现在给定一个矩形区域 `[L, R] × [D, U]`，要求统计其中所有黑色整点的数量。

## 解题思路
题目的染色规则取决于 `max(|x|, |y|)` 的奇偶性，因此关键是先判断这个最大值到底由谁提供。

我们可以把所有整点分成两类讨论：

1. 当 `|x| > |y|` 时，此时 `max(|x|, |y|) = |x|`。
   如果该点是黑色，就要求 `|x|` 为偶数，而一个数与它的绝对值奇偶性相同，所以等价于 `x` 为偶数。

   因此代码第一部分直接枚举区间 `[L, R]` 中的每个 `x`，只处理偶数的 `x`。  
   对于固定的 `x`，还需要满足 `|x| > |y|`，也就是：
   `-|x| < y < |x|`

   因为 `y` 是整数，上式可以改写成：
   `-|x| + 1 <= y <= |x| - 1`

   再和题目给出的纵坐标范围 `[D, U]` 取交集，得到：
   `dd = max(D, -abs(x) + 1)`
   `uu = min(U, abs(x) - 1)`

   如果 `dd <= uu`，说明这个 `x` 能贡献 `uu - dd + 1` 个黑点，这正对应代码中的：
   `res += max(uu - dd + 1, 0);`

2. 当 `|x| <= |y|` 时，此时 `max(|x|, |y|) = |y|`。
   同理，黑色条件变成 `|y|` 为偶数，也就是 `y` 为偶数。

   所以代码第二部分枚举区间 `[D, U]` 中的每个偶数 `y`。  
   对于固定的 `y`，要满足 `|x| <= |y|`，即：
   `-|y| <= x <= |y|`

   再和横坐标范围 `[L, R]` 取交集，得到：
   `ll = max(L, -abs(y))`
   `rr = min(R, abs(y))`

   如果 `ll <= rr`，那么这个 `y` 能贡献 `rr - ll + 1` 个黑点，对应代码中的：
   `res += max(rr - ll + 1, 0);`

这两类情况正好覆盖了所有整点，并且不会重复统计，因此把两部分答案相加就是最终结果。

时间复杂度为 `O((R - L + 1) + (U - D + 1))`，因为只需要各扫一遍横坐标和纵坐标区间。答案数量可能较大，所以代码使用 `long long` 保存结果。

## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
ll res;
int l, r, d, u;
int main(){
    scanf("%d%d%d%d", &l, &r, &d, &u);
    // |x| > |y|
    for(int x=l;x<=r;x++){
        if(x%2==0){
            int dd=max(d, -abs(x)+1);
            int uu=min(u, abs(x)-1);
            res+=max(uu-dd+1, 0);
        }
    }
    // |x| <= |y|
    for(int y=d;y<=u;y++){
        if(y%2==0){
            int ll=max(l, -abs(y));
            int rr=min(r, abs(y));
            res+=max(rr-ll+1, 0);
        }
    }
    printf("%lld\n", res);
    return 0;
}
```
