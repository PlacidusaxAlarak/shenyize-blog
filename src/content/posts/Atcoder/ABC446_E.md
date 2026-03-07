---
title: "AtCoder ABC446_E: E - Multiple-Free Sequences
			Editorial"
published: 2026-03-07
description: "AtCoder 算法题解：E - Multiple-Free Sequences
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc446/tasks/abc446_e](https://atcoder.jp/contests/abc446/tasks/abc446_e)

## 题目大意

给定三个整数 `M, A, B`。

对于任意初始二元组 `(x, y)`，其中 `0 <= x, y < M`，构造无限序列：

- `s_1 = x`
- `s_2 = y`
- `s_n = A s_{n-1} + B s_{n-2}`（`n >= 3`）

要求统计有多少组 `(x, y)`，能使这个无限序列中 **不存在任何一项是 `M` 的倍数**。

也就是说，序列里不能出现满足 `s_n mod M = 0` 的项。

输出满足条件的初始二元组数量。

## 解题思路

先把问题放到模 `M` 的意义下来看。

因为我们只关心某一项是否是 `M` 的倍数，所以只需要考虑每一项对 `M` 取模后的结果。于是每个数都可以看成 `0 ~ M-1` 之间的值。

设当前连续两项为 `(u, v)`，那么下一项就是：

- `next = (A * v + B * u) mod M`

因此状态 `(u, v)` 会转移到：

- `(v, (A * v + B * u) mod M)`

也就是说，我们可以把“连续两项”当成图上的一个点。由于 `u, v` 都只有 `M` 种取值，所以总状态数只有 `M^2` 个。

如果某个状态中有一项为 `0`，那么就说明序列里已经出现了 `M` 的倍数，因此这类状态都是**坏状态**。代码里直接把 `(u, v)` 中满足 `u = 0` 或 `v = 0` 的状态全部作为起点。

题目要求的是：有多少个初始状态 `(x, y)`，在不断转移后**永远不会走到坏状态**。

正向去判断每个起点会不会走到坏状态不太方便，但这个图的转移关系很规整，所以更适合**反向建图**。

如果正向有：

- `(i, j) -> (j, (A * j + B * i) mod M)`

那么反向边就是：

- `(j, (A * j + B * i) mod M) -> (i, j)`

这正对应代码中的：

```cpp
ll k = (i * b + j * a) % m;
g[j * m + k].push_back(i * m + j);
```

接下来从所有坏状态出发，在反图上做一遍 BFS。这样所有能够被访问到的状态，都是“最终能走到坏状态”的状态，也就是不合法的初始状态；而没有被访问到的状态，就是答案。

最后统计所有没有被标记的 `(x, y)` 个数即可。

### 复杂度分析

- 状态数为 `M^2`
- 每个状态恰好连出一条正向边，因此反图边数也是 `M^2`

所以总时间复杂度为 `O(M^2)`，空间复杂度也为 `O(M^2)`。

## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=1010;
typedef long long ll;
ll m, a, b;
vector<ll> g[N*N];//建图
queue<ll> q;
bool st[N*N];//判断是不是会达到0的点
int main(){
    scanf("%lld%lld%lld", &m, &a, &b);
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            //(i, j)<-(j, i*b+j*a)
            ll k=(i*b+j*a)%m;//反向建图
            g[j*m+k].push_back(i*m+j);//将二维的点压成一维(i, j)->i*m+j
        }
    }
    for(ll i=0;i<m;i++){
        for(ll j=0;j<m;j++){
            if(i==0||j==0){//凡是带0的, 都是不行的点, 都要计算其位置
                ll v=i*m+j;
                st[v]=true;
                q.push(v);//多源bfs
            }
        }
    }
    while(!q.empty()){
        ll t=q.front();
        q.pop();
        for(ll u:g[t]){
            if(!st[u]){//凡是被bfs遍历到的, 都是能变成%M==0的点, 都会被加入
                st[u]=true;
                q.push(u);
            }
        }
    }
    ll res=0;
    for(ll i=0;i<m;i++){
        for(ll j=0;j<m;j++){
            if(st[i*m+j]==0) res++;//st值为false的, 是答案, 要++
        }
    }
    printf("%lld\n", res);
    return 0;
}
```
