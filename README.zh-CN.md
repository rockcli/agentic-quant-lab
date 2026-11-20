# agentic-quant-lab

[![CI](https://github.com/rockcli/agentic-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rockcli/agentic-quant-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rockcli/agentic-quant-lab?display_name=tag&color=10B981)](https://github.com/rockcli/agentic-quant-lab/releases/tag/v0.1.0)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

**面向知识库增强（RAG）的智能体量化策略研究与反过拟合验证框架。**

`agentic-quant-lab`（包名 `agentic_quant`）将本地 RAG 驱动的 LLM 策略研究闭环（生成 -> 回测 -> 诊断 -> 迭代 -> 实验追踪）与统计学反过拟合验证套件（CSCV/PBO、Deflated Sharpe Ratio、带 Purge Gap 的步进检验 WFA、四象限市场状态划分、样本外 Embargo 跟踪）结合在一起。

行情抓取与实盘交易接口已**彻底解耦**：所有流水线、回测引擎与验证门禁均只接收调用方显式传入的 `Bar` 列表、NumPy OHLCV 数组或自定义 `fetch_func` 回调。

---

## 整体架构

```
+-------------------------------------------------------------------------+
|                       1. 本地知识库检索 (RAG)                           |
|  docs/knowledge_base/*.md                                               |
|    -> MarkdownChunker -> TfidfEmbedder / SentenceTransformersEmbedder   |
|    -> InMemoryVectorStore -> HybridRetriever / BookRetriever            |
+------------------------------------+------------------------------------+
                                     | 检索理论片段
                                     v
+-------------------------------------------------------------------------+
|                       2. 智能体策略研究闭环                             |
|  ResearchOrchestrator                                                   |
|    -> StrategyGenerationPipeline (自然语言 -> AST 校验策略代码)         |
|    -> BacktestEngine (事件驱动回测，支持 close / next_open 撮合)       |
|    -> DiagnosisPipeline (结合 AFML 原则的回撤与左尾风险诊断)            |
|    -> AutoIterationPipeline <-> ExperimentTracker (JSONL 实验谱系记录)  |
+------------------------------------+------------------------------------+
                                     | 候选策略与参数网格
                                     v
+-------------------------------------------------------------------------+
|                      3. 反过拟合统计验证门禁                            |
|  +-----------------------+  +--------------------+  +-----------------+ |
|  | 过拟合检测            |  | 净化步进检验 (WFA) |  | Embargo 与状态  | |
|  | - CSCV / PBO          |  | - Purge Gap 隔离   |  | - 趋势x波动率   | |
|  | - Deflated Sharpe     |  | - 多数原则 (>=70%) |  |   四象限划分    | |
|  | - 参数高原与悬崖检测  |  | - 极端回撤一票否决 |  | - 样本外跟踪    | |
|  +-----------------------+  +--------------------+  +-----------------+ |
+-------------------------------------------------------------------------+
```

---

## 核心模块

### 1. 智能体研究闭环 (`agentic_quant.ai`)
- **轻量 LLM 客户端 (`LLMClient`)**：基于标准库 `urllib` 实现，兼容 OpenAI / DeepSeek / Kimi 接口格式（通过 `DEFAULT_LLM_PROVIDER`、`<PROVIDER>_API_KEY`、`<PROVIDER>_BASE_URL` 配置），内置重试、Token 统计与 SHA-256 本地缓存（24 小时有效期）。
- **本地混合检索 (`agentic_quant.ai.rag`)**：
  - `MarkdownChunker` + `TfidfEmbedder`（零外部深度学习依赖）+ `InMemoryVectorStore` + `HybridRetriever`（语义与关键词混合检索，可选 `SentenceTransformersEmbedder`）。
  - `BookRetriever`：面向 `docs/knowledge_base/` 的轻量 BM25 风格本地检索器。
- **端到端编排 (`ResearchOrchestrator`)**：
  - `StrategyGenerationPipeline`：检索知识片段、拼装 Prompt、提取代码并做 `ast.parse` 与 `Strategy` 子类检查。
  - `DiagnosisPipeline`：从 `BacktestResult` 提取关键指标、亏损样本与回撤区间，输出结构化 `DiagnosisReport`。
  - `AutoIterationPipeline` 与 `ExperimentTracker`：自动调参重测并持久化父子实验谱系（`records.jsonl`）。

### 2. 反过拟合验证套件 (`agentic_quant.validation` & `agentic_quant.optimization`)
- **CSCV 与回测过拟合概率 (`overfit.py`)**：
  - `cscv_pbo`：按时间块切分并保持块内时序自相关结构的组合对称交叉验证，输出 `PBOResult`（PBO、Logit 分布、IS->OOS 衰减率、Deflated Sharpe Ratio 及 `A`/`B`/`C`/`D` 评级）。
  - `ParameterStabilityAnalyzer`：一维及网格参数高原宽度计算与相邻参数悬崖检测（下跌超过 50%）。
  - `run_pbo_single` 与 `run_pbo_batch`：基于 OHLCV 数组或显式 `fetch_func` 执行单标的/批量 PBO 评估。
- **净化步进检验 (`walk_forward.py`)**：
  - `WalkForwardOptimizer` 与 `WalkForwardAnalyzer`：在训练集与测试集之间加入 **Purge Gap** 防止标签泄漏，结合**多数原则**（>=70% OOS 窗口正收益）、**灾难性回撤一票否决**、参数稳定性与分状态评估生成 Markdown/JSON 报告。
- **四象限市场状态分类 (`regime.py`)**：
  - `MarketRegime`：基于 MA(200) + 滚动线性斜率与 20 日波动率历史中位数，将市场划分为 `bull_high_vol`、`bull_low_vol`、`bear_high_vol`、`bear_low_vol`。
- **样本外 Embargo 跟踪 (`embargo.py`)**：
  - `EmbargoManager`：记录回测结束后的样本外信号并逐日核验真实行情、跟踪累计盈亏与止损/单日/累计熔断告警。
- **纯 NumPy 信号引擎与风险指标 (`signal_engine.py`, `labeling.py`, `vpa.py`, `weekly.py`, `risk/metrics.py`)**：
  - 向量化信号（`sma`、`kama`、`donchian`、`donchian_regime`、`bollinger`）、量价确认（VPA）、三屏障标签（`triple_barrier_labels`）以及非正态夏普修正指标（`probabilistic_sharpe`、`deflated_sharpe`、`calmar_ratio`、`sortino_ratio`、`omega_ratio`）。

### 3. 事件驱动回测与示例策略 (`agentic_quant.backtest`, `agentic_quant.strategy`)
- `BacktestEngine`：支持 `execution_timing="close"` 与 `"next_open"`（避免未来函数）及手续费/滑点模拟，通过 Polars 输出 `BacktestResult`。
- `agentic_quant.strategy.examples` 提供通用教科书示例策略：`SmaCrossStrategy`、`DonchianBreakout`、`BollingerMeanReversion`、`PairsTrading`、`VolumeProfileStrategy`、`MetaLabelingVpaStrategy`。

---

## 安装与运行测试

需要 Python `>=3.11`，核心运行时依赖仅需 `numpy` 与 `polars`：

```bash
# 安装开发依赖（推荐使用 uv）
uv sync --extra dev

# 运行全量离线测试
uv run --with pytest --with numpy --with polars pytest
```

所有测试均基于合成行情与模拟 LLM 响应，可在完全离线环境下运行。

---

## 非目标

- **非实盘交易、券商下单或行情源工具** — 外部行情与执行完全解耦（向 `run_pbo_batch` 传入 `fetch_func=None` 会显式抛出 `ValueError`）。
- **不自动沙盒执行 LLM 生成的策略代码** — `ResearchOrchestrator.run()` 刻意不 `exec()` 生成的策略文件（默认运行占位策略 `SmaCrossStrategy`，需人工审查代码后再接入验证链路）。
- **不做高频逐笔订单簿或融资融券模拟** — `BacktestEngine` 面向单资产 K 线（固定费率 + 加法滑点），不模拟盘口深度、部分成交或借券/资金费率。
- **非持久化向量数据库** — `InMemoryVectorStore` 仅在进程内存中维护 NumPy 向量，每次启动会重建 TF-IDF 索引。
- **不构成任何投资建议** — 统计门禁只能降低历史样本过拟合概率，无法保证未来实盘收益。

## 已知坑点

- **默认 `close` 撮合与严格 T+1 隔离** — 为兼容既有接口，`BacktestEngine` 及 WFA/网格/自动迭代链路默认仍使用 `execution_timing="close"`；严格次日开盘成交、剔除隔夜跳空泄漏的实现位于 `validation/signal_engine.py`（`run_strategy()`）。
- **CSCV 采样与无方向 DSR** — `cscv_pbo()` 采用蒙特卡洛分块抽样而非全量 `C(S, S/2)` 枚举；其中 `_compute_deflated_sharpe()` 使用了 `abs(mean_logit)`，导致稳定过拟合的策略（`logit > 0`）也会拿到高 `dsr`，必须与 `pbo` 联合判读。
- **辅助函数边界与短窗口吞没** — `apply_weekly_filter()` 默认将每周最后一个交易日收盘价广播至当周周一至周五（实盘化需额外滞后 1 周）；`MarketRegime.classify()` 使用全样本波动率中位数仅适合事后 WFA 归因（实时因果特征请改用 `classify_regime_percentile()`）；短周期日内窗口若小于默认 `purge_days=5` 会被直接跳过。

## 安全与信任边界

- **核心包零动态代码执行** — `src/agentic_quant/` 内无任何 `subprocess`、`eval`、`exec` 或 `pickle` 调用；生成的 `.py` 策略仅做 `ast.parse` 语法与结构校验，导入执行前必须人工审计。
- **受控出网与明文缓存** — 仅 `LLMClient.chat()` 会通过标准库 `urllib.request.urlopen` 请求 `<PROVIDER>_BASE_URL/chat/completions`（携带 Bearer Token）；`base_url` 未做协议/域名白名单限制，必须视为受信任配置。提示词与模型响应会明文缓存在 `data/cache/llm_cache/`（TTL 24 小时，16 位十六进制 SHA-256 摘要命名）。
- **本地可信输入假设** — `docs/knowledge_base/*.md` 内容会原样拼入 RAG 提示词（无提示词注入清洗），`StrategyGenerationPipeline.save()` 亦直接基于截取的 `description` 生成文件名，两者均假设为可信本地输入。

---

## 方法论说明

- 本仓库提供的是**策略研究与统计防过拟合验证工具**，不包含任何未经验证的实盘收益率承诺。
- 由于实盘/第三方数据源已解耦，向 `run_pbo_batch` 传入 `fetch_func=None` 会显式抛出 `ValueError`。

## 开源协议

MIT © [rockcli](LICENSE)
