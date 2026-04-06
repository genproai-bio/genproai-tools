# GenProAI Tools 方法学论文

## 目标
Bioinformatics Application Note (2 pages) 或 Briefings in Bioinformatics

## 暂定标题
GenProAI Tools: A unified Python toolkit for multi-dimensional tumor microenvironment characterization

## 核心卖点
1. 7 个高水平论文算法的统一 Python 重实现，零新环境依赖
2. 修复原版已知 bug（Shears 6 个 bug 全部修复，含对照验证）
3. 组合分析产出新维度（三层代谢分析、senescence × spatial 等）
4. 实际项目验证（OV-2 + STAD-1 + 后续项目的数据）

## 目录结构

```
methods_paper/
├── README.md          ← 本文件
├── manuscript.md      ← 正文草稿 (待写)
├── data/              ← 验证用数据 / benchmark 结果
├── figures/           ← 论文图表
└── scripts/           ← 生成图表的脚本
```

## 所需素材 (待积累)
- [ ] 7 模块 vs 原版的 benchmark 对比 (准确性 + 速度 + 依赖数)
- [ ] ≥2 个瘤种的实际应用案例 (OV-2 ✅, STAD-1 待跑)
- [ ] 组合分析 showcase (三层代谢 ✅, cytotrace+senescence 待跑)
- [ ] 依赖环境对比图 (7 个原版 env vs 1 个 bioinfo env)

## 作者
基智远团队（不需要临床合作者）

## 时间线
Phase 1 验证数据就绪后 2 个月内投稿
