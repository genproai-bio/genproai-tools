"""
GenProAI Tools — 基智远内部生物信息分析工具包

从高水平论文的开源实现中提取核心算法，在统一环境下重实现。
避免依赖地狱，保留算法精髓，支持业务定制。

Modules:
    spatial              - 空间蛋白组/转录组交互分析 (源自 Scimap)
    senescence           - 衰老细胞检测 (源自 DeepScence)
    metabolic            - 代谢通量推断 (源自 METAFlux)
    cytotrace            - 细胞分化潜能评分 (源自 CytoTRACE)
    stable_selection     - 稳定特征选择 with FDR (源自 Stabl)
    bias_detection       - biomarker 预测偏差检测 (源自 HistBiases)
    phenotype_association - bulk-scRNA 表型关联 (源自 Shears, 修6个bug)
    drug_sensitivity     - 单细胞药物敏感性预测 (源自 DREEP, gf-icf + rank enrichment)
    cin_signature        - 17 CIN signatures 量化 (源自 CINSignatureQuantification)
    metastasis_index     - 原发灶 vs 转移灶 TME 重塑指标 RI/TI (源自 PBMA)
    covarying_neighborhoods - 协变量关联细胞状态变化检测 (源自 rcna)
"""

__version__ = "0.4.0"
