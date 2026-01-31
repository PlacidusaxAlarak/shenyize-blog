---
title: "AtCoder ABC442_E: E - Laser Takahashi
			Editorial"
published: 2026-01-31
description: "AtCoder 算法题解：E - Laser Takahashi
			Editorial"
tags: ["AtCoder", "算法"]
category: "算法"
priority: 0
---

## 题目链接
[https://atcoder.jp/contests/abc442/tasks/abc442_e](https://atcoder.jp/contests/abc442/tasks/abc442_e)

## 题目大意
### 问题描述
在二维平面上有 $N$ 个怪兽，第 $i$ 个怪兽的坐标为 $(X_i, Y_i)$（坐标不为原点）。
高桥位于原点 $(0,0)$。他的眼睛会发射强力激光，能瞬间消灭**当前面朝方向射线上**的所有怪兽。

### 询问
青木进行了 $Q$ 次独立的思想实验。对于第 $j$ 次实验，给定两个怪兽编号 $A_j$ 和 $B_j$：
1.  初始时，高桥面朝怪兽 $A_j$ 的方向。
2.  高桥开始**顺时针**旋转身体。
3.  当他面朝怪兽 $B_j$ 的方向时，立即停止旋转。

**目标：** 计算在整个旋转过程中（包含起始方向和结束方向），总共有多少个怪兽被消灭。

### 注意事项
* 如果在某一方向上存在多个怪兽，它们会被同时消灭。
* 如果怪兽 $A_j$ 和 $B_j$ 相对于原点在同一方向上，高桥不会旋转，仅消灭该方向上的怪兽。
* $N, Q \le 2 \times 10^5$，坐标范围 $\pm 10^9$。
## 解题思路
本题本质是**环形区间的区间求和**。

1.  **极角排序 (转化)**
    将二维坐标转化为一维的顺时针顺序。
    * **方法**：将平面按顺时针划分为 8 个区域（轴+象限），区域内利用**叉积**判断先后。
2.  **离散化分组 (去重)**
    排序后，同一射线上的怪兽会相邻。将它们合并为同一个 `group_id`，记录该组的怪兽数量 `cnt`。
3.  **前缀和 (查询)**
    对 `cnt` 数组做前缀和。
    * 若 $A \to B$ 未跨越起跑线：`sum[v] - sum[u-1]`。
    * 若跨越起跑线（转了一圈）：`(sum[Total] - sum[u-1]) + sum[v]`。
## 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+10;
int n, q;
struct point{
	int idx, quadrant, groupid;
	long long x, y;
}a[N];
int idgroup[N], groupcnt[N], sum[N];
int u, v;
long long cross(point a, point b){
	return a.x*b.y-a.y*b.x;
}
bool cmp(point a, point b){
	if(a.quadrant!=b.quadrant) return a.quadrant<b.quadrant;
	long long cp=cross(a, b);
	return cp<0;
}
int main(){
	scanf("%d%d", &n, &q);
	for(int i=1;i<=n;i++) {
		scanf("%lld%lld", &a[i].x, &a[i].y);
		a[i].idx=i;
		if(a[i].x==0&&a[i].y>0) a[i].quadrant=0;//象限 
		if(a[i].x>0&&a[i].y>0) a[i].quadrant=1;
		if(a[i].x>0&&a[i].y==0) a[i].quadrant=2;
		if(a[i].x>0&&a[i].y<0) a[i].quadrant=3;
		if(a[i].x==0&&a[i].y<0) a[i].quadrant=4;
		if(a[i].x<0&&a[i].y<0) a[i].quadrant=5;
		if(a[i].x<0&&a[i].y==0) a[i].quadrant=6;
		if(a[i].x<0&&a[i].y>0) a[i].quadrant=7;
	}
	sort(a+1, a+n+1, cmp);
	int m=0;
	for(int i=1;i<=n;i++) {
		if(i==1||a[i].quadrant!=a[i-1].quadrant||cross(a[i], a[i-1])!=0){
			m++;
			groupcnt[m]=0;
		}
		a[i].groupid=m;
		idgroup[a[i].idx]=m;
		groupcnt[m]++;
	}
	for(int i=1;i<=m;i++) sum[i]=sum[i-1]+groupcnt[i];
	while(q--){
		scanf("%d%d", &u, &v);
		int startg=idgroup[u];
		int endg=idgroup[v];
		if(startg==endg) printf("%d\n", groupcnt[startg]);
		else{
			int ans=0;
			if(startg<endg){
				ans=sum[endg]-sum[startg-1];
			}
			else {
				ans+=sum[m]-sum[startg-1];
                ans+=sum[endg];
			}
			printf("%d\n", ans);
		}
	}
	return 0;
}
```
> "千里之行，始于足下。"