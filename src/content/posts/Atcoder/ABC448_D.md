---
title: "AtCoder ABC448_D: D - Integer-duplicated Path
			Editorial"
published: 2026-03-12
description: "AtCoder 算法题解：D - Integer-duplicated Path
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc448/tasks/abc448_d](https://atcoder.jp/contests/abc448/tasks/abc448_d)

## 题目大意
    给定一棵有 `N` 个节点的树，第 `i` 个节点上写着一个整数 `A_i`。

对于每个 `k=1,2,\dots,N`，考虑树上从节点 `1` 到节点 `k` 的唯一路径，判断这条路径上是否存在两个不同节点写着相同的整数。

如果存在，输出 `Yes`；否则输出 `No`。

## 解题思路
    如果用BFS来为每个节点维护一个set集合的话, 毫无疑问MLE, 所以我们要使用DFS, 对树上的点进行遍历, 具体做法就是到达一个点, 先判断他的父节点是不是就已经重复了, 如果已经重复了, 就直接标记即可, 如果没有重复, 用unordered_map统计这个点的值出现的次数, 判断有没有重复, 再进行下一次的DFS, 然后回溯的时候记得在map上减掉这个值的数量即可
## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+10;
int a[N];
vector<int> q[N];
bool success[N];
unordered_map<int, int> m;
int n, u, v;
void dfs(int u, int fa){
    for(int t:q[u]){
        if(t==fa) continue;
        m[a[t]]++;
        if(success[u]) success[t]=true;
        if(m[a[t]]>=2) success[t]=true;
        dfs(t, u);
        m[a[t]]--;
    }
}
int main(){
    scanf("%d", &n);
    for(int i=1;i<=n;i++) scanf("%d", &a[i]);
    for(int i=1;i<=n-1;i++){
        scanf("%d%d", &u, &v);
        q[u].push_back(v), q[v].push_back(u);
    }
    m[a[1]]++;
    dfs(1, -1);
    for(int i=1;i<=n;i++){
        if(success[i]==false) printf("No\n");
        else printf("Yes\n");
    }
    return 0;
}
```
