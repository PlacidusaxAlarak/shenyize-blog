---
title: "AtCoder ABC447_E: E - Divide Graph
			Editorial"
published: 2026-03-02
description: "AtCoder 算法题解：E - Divide Graph
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc447/tasks/abc447_e](https://atcoder.jp/contests/abc447/tasks/abc447_e)

## 题目大意


**背景设定**
给定一个包含 $N$ 个顶点和 $M$ 条边的简单连通无向图。
每条边都有一个特定的“成本”：第 $i$ 条边的成本为 $2^i$。

**操作目标**
你需要从图中选择并删除一部分边，使得原图被**恰好划分为 2 个连通分量**（即把整个图一分为二）。

**求解问题**
在所有满足条件的划分方案中，找到**被删除边的成本总和的最小可能值**。
最终的结果需要对 $998244353$ 取模后输出。

---

**核心要点提醒：**
* 必须是“恰好 2 个连通分量”。
* 边权具有 $2^i$ 的特殊性质，这意味着保留一条大边总是优于保留所有比它小的边之和（$2^i > \sum_{j=1}^{i-1} 2^j$）。
* 注意：是求出真实的“最小成本总和”之后，再去对 $998244353$ 取模，而不是在取模的意义下比大小。
## 解题思路
    注意到权值是 2^i, 容易想到 从1到i-1条边的权值加起来都没有第i条边的一个权值高这个二进制的性质, 于是我们采用"正难则反"的思路解决问题:要求最小删除的边的权值和,即保留最大的边权值和。
    题目的要求是缩边, 根据正难则反的思路, 我们考虑增边。即假设一开始的点都是相互之间孤立的, 我们不断将孤立的点连接起来，组成一个个连通块，当连通块数量达到2时，就可以停止，以计算需要删除的边的权值
    我们仿照"最小生成树"的思路, 实现"最大生成树", 并使用并查集进行判断是否在同一个连通块中, 具体逻辑请见代码及其注释

## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+10;
typedef long long ll;
const ll mod=998244353ll; //取模的值
int n, m;
vector<pair<int, int>> g;//存边的两点的vector
int fa[N];//并查集所需数组
ll res;
int find(int x){//路径压缩+查询
    if(x!=fa[x]) return fa[x]=find(fa[x]);
    return x;
}

int main(){
    scanf("%d%d", &n, &m);
    int cnt=n;//一开始连通块数量为n
    for(int i=0;i<m;i++) {
        int u, v;
        scanf("%d%d", &u, &v);
        g.push_back({u, v});
    }
    for(int i=1;i<=n;i++) fa[i]=i;//初始化并查集信息
    for(int i=m-1;i>=0;i--){//倒序, 从最大的开始
        if(cnt==2) break;//只剩两个了, 就可以停止, 因为再合并下去就不满足两个联通分量的要求了. 剩下的边要不属于连通分量1, 要不属于连通分量2, 要不是横跨两个连通分量
        int u=g[i].first, v=g[i].second;
        u=find(u), v=find(v);
        if(u==v) continue;
        fa[u]=v;//合并
        cnt--; //连通块数量-1
    }
    ll p=1;
    for(int i=0;i<m;i++){
        p=p*2%mod;//p就是每个边的权值%mod
        int u=g[i].first, v=g[i].second;
        if(find(u)==find(v)) continue;//是同一个连通分量里的(要不是1, 要不是2)
        res=(res+p)%mod;//这个是横跨了两个分量的, 一定要删掉
    }
    printf("%lld\n", res);
    return 0;
}
```
