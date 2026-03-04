---
title: "AtCoder ABC447_D: D - Take ABC 2
			Editorial"
published: 2026-03-01
description: "AtCoder 算法题解：D - Take ABC 2
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc447/tasks/abc447_d](https://atcoder.jp/contests/abc447/tasks/abc447_d)

## 题目大意


给定一个长度不超过 $10^6$ 且仅由字符 `A`、`B` 和 `C` 组成的字符串 $S$。

每次操作你可以执行以下步骤：
1. 在字符串中找到一个（不一定连续的）子序列 `ABC`。也就是说，找到三个下标 $i, j, k$ 满足 $1 \le i < j < k \le |S|$，并且 $S_i = \text{A}, S_j = \text{B}, S_k = \text{C}$。
2. 将这三个匹配的字符从原始字符串中删除。
3. 剩下的字符按原本的相对顺序向左靠拢，重新拼接成一个新的字符串。

**目标**：求出对该字符串最多可以执行多少次上述操作。
## 解题思路
核心观察：每次删除 `ABC` 后，剩余字符的相对顺序不变，因此能形成 `ABC` 的三元组只取决于原串的位置关系。问题等价于在原串中选出最多个不相交的 `(A, B, C)`，满足索引严格递增。

贪心策略：总是使用最靠左的可行三元组。对每个 `A`，匹配它右侧最早的 `B`，再匹配该 `B` 右侧最早的 `C`。选择越靠左，给后面留下的可选字符越多，因此不会降低最优解。

实现方式：预处理 `A/B/C` 的位置数组，三指针线性扫描。
1. `i` 从左到右枚举 `A`。
2. `j` 向右移动到第一个满足 `B[j] > A[i]` 的位置。
3. `k` 向右移动到第一个满足 `C[k] > B[j]` 的位置。
4. 若 `j` 或 `k` 到末尾则结束；否则计数并 `j++、k++` 继续。

时间复杂度 `O(N)`，空间复杂度 `O(N)`（存位置数组）。
## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=1e6+10;
int a[N], b[N], c[N]; //存储A, B, C字符的位置
int idxa, idxb, idxc, res;//记录A, B, C有多少个
string s;
int main(){
    cin>>s;
    int n=s.size();
    for(int i=0;i<n;i++) {
        if(s[i]=='A') a[idxa++]=i;
        if(s[i]=='B') b[idxb++]=i;
        if(s[i]=='C') c[idxc++]=i;
    }//记录 A, B, C字符的位置
    int j=0, k=0;//当前B和C的位置
    for(int i=0;i<idxa;i++){
        while(b[j]<a[i]&&j<idxb) j++;//如果目前B的位置在A面前, 就要++, 直到要在B的后面
        while((c[k]<b[j]||c[k]<a[i])&&k<idxc) k++;//同理, 要在A和B的后面, 也要加上<idxc的限制
        if(j==idxb||k==idxc) break;//到头了, 就说明没有再符合的了
        res++;//没执行上一个语句, 就说明找到一个合理的
        j++, k++;//j和k都要移向下一位, 同时i++
    }
    printf("%d\n", res);
    return 0;
}
```
