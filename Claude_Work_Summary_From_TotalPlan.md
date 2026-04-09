# Claude Work Summary (From Total Plan)

## 概述

同事使用 Claude Code (Sonnet 4.6 / Opus 4.6) 在 `shenhao@10.176.55.210` 远程服务器上对 ScholarGym 项目进行了两大阶段的工作：**代码审查与重构规划** 和 **NeurIPS 投稿准备**。工作时间大约在 2026-03-28。

---

## 第一阶段：代码审查与重构规划

### 已完成

1. **生成 CLAUDE.md**：项目概述、运行命令、架构说明、LLM 配置和关键注意事项。

2. **代码审查（评分 5.5/10）**，识别出以下问题：
   - ~930 行重复代码（prompt.py 480行、eval.py 140行、api.py 70行、build脚本 167行、Agent样板 80行）
   - 扁平配置系统（无类型、无验证）
   - `utils.py` 混杂 5 种不相关职责
   - 零测试覆盖
   - RAG 系统中 GT 数据泄漏到检索逻辑（后确认仅用于 debug，不影响结果）
   - 各模块评分：api.py 7/10, config.py 4/10, eval.py 5/10, prompt.py 4/10, utils.py 4/10, rag.py 5.5/10

3. **5 阶段重构方案**（未执行，仅规划）：
   - Phase 0: `scholargym/` 包结构 + Pydantic Settings 配置
   - Phase 1: 消除重复代码（prompt 参数化、API 统一、build 去重、Agent 基类）
   - Phase 2: 拆分 utils.py、分离 GT 与检索、数据模型清理
   - Phase 3: 依赖注入、Workflow 基类、状态封装
   - Phase 4: 测试基础设施、错误处理、requirements 清理
   - Phase 5: CLI 提取、包最终化

### 未完成

- **整个重构方案均未执行**，仅停留在规划阶段，用户批准了计划后转向了 NeurIPS 投稿任务。

---

## 第二阶段：NeurIPS 2026 投稿准备

### 已完成

1. **API 配置更新** (`code/api.py`)
   - 新增 `claude` provider（113.45.39.247:3001）
   - 新增 `minimax` provider（api.scnet.cn）
   - 更新 `gpt` provider 指向 ohmyapi（gpt-5.4）
   - 更新 `glm` provider 的 zhipu API key

2. **Metrics 修复** (`code/metrics.py`, `code/utils.py`, `code/eval.py`)
   - 新增 F1 计算（retrieval_f1 + selection_f1）
   - 修复 `avg_distance` 除零错误（line 272）
   - 新增 `gt_discard_rate` 字段
   - 简化 eval.py 中 7 个重复的 elif 聚合块为单循环

3. **Config Loader 修复** (`code/eval.py`)
   - 修复动态 config 加载时 `from config import *` 的循环引用问题
   - 自定义 config 文件（如 `config_gpt54.py`）现在能正确继承默认配置

4. **新建模型配置文件**
   - `code/config_gpt54.py` — GPT-5.4
   - `code/config_claude_sonnet.py` — Claude Sonnet 4.6
   - `code/config_minimax.py` — MiniMax M2.5
   - `code/config_glm5.py` — GLM-5

5. **数据传输**（从 10.176.55.210 SCP 到本地）
   - ICML 论文源文件 → `paper/ICML/`
   - 历史评估结果 → `eval_final/`
   - 分析脚本 → `code/plot_evaluation_summary.py`, `fix_closed_model_metrics.py`, `estimate_closed_model_metrics.py`
   - 论文数据库 `scholargym_paper_db.json`（820MB）
   - BM25 索引 `bm25_index.pkl`（2GB）
   - 实验子集 `superlong_bench_300.jsonl`（300 queries）

6. **NeurIPS 论文转换** (`paper/neurips2026/`)
   - `main.tex` — Main Track, double-blind, clean structure
   - 5 个 section 文件从 ICML 格式转换
   - 移除 findingbox（tcolorbox）
   - `table*`/`figure*` → 单栏 `table`/`figure`
   - 移除 ICML 特有宏（`\icmltitle`, `\icmlauthor` 等）
   - 填写 NeurIPS checklist（16 项）
   - 去除 debug 宏（`\gzh`, `\blue`, `todonotes`）

7. **GPT-5.4 Pipeline 验证**
   - 10 条 query / 2 iterations 验证通过（22分钟）
   - Checkpoint resume 功能正常

8. **API 连通性验证**
   - GPT-5.4（ohmyapi）：**通过**
   - Claude Sonnet 4.6（113.45.39.247）：**通过**
   - MiniMax M2.5（scnet）：**失败** — 402 余额不足
   - GLM-5（zhipu）：**失败** — 401 API key 过期

9. **第二轮 Code Review**（更深入），发现：
   - **Critical**: `deeprag.py:364` ID 类型不匹配（p.id vs p.arxiv_id）
   - **Critical**: `rag.py` BM25 和 vector 的 rank 计算不一致
   - **Critical**: `utils.py:36` 正则贪婪匹配导致回溯
   - **High**: checkpoint rebuild 缺少 F1 指标
   - **High**: prompt 模板无 XML 转义
   - **High**: selector 返回类型声明不匹配
   - 以及 7 个 MEDIUM 级问题

10. **GPT-5.4 全量实验已启动**
    - 数据集: `superlong_bench_300.jsonl`（300 queries, 5 iterations）
    - PID: 3576046
    - 预估 ~10 小时完成
    - 结果目录: `eval_results/gpt-5.4_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/`

### 未完成

1. **MiniMax M2.5 实验** — scnet 账户余额不足，需充值
2. **GLM-5 实验** — zhipu API key 过期，需更新
3. **Claude Sonnet 4.6 实验** — API 已通但实验未启动
4. **GPT-5.4 全量实验结果收集** — 正在后台运行中
5. **论文表格更新** — 需等新模型实验完成后，更新 Table 1-4 的数据
6. **论文写作润色** — 减少 AI 风格总结性语句，使用地道学术英语
7. **论文编译验证** — 本地和远程均未安装 LaTeX，未验证 NeurIPS 版本是否能编译
8. **第二轮 Code Review 中发现的 Critical/High 问题均未修复**
9. **重构方案完全未执行**（Phase 0-5）

---

## 关键文件变更清单

| 文件 | 状态 | 变更内容 |
|------|------|----------|
| `code/api.py` | 已修改 | 新增 claude/minimax provider，更新 gpt/glm 默认值 |
| `code/eval.py` | 已修改 | config loader 修复，F1 收集，聚合简化 |
| `code/metrics.py` | 已修改 | F1 计算，gt_discard_rate，除零保护 |
| `code/utils.py` | 已修改 | retrieval_f1 + selection f1 |
| `code/config_gpt54.py` | 新建 | GPT-5.4 实验配置 |
| `code/config_claude_sonnet.py` | 新建 | Claude Sonnet 配置 |
| `code/config_minimax.py` | 新建 | MiniMax 配置 |
| `code/config_glm5.py` | 新建 | GLM-5 配置 |
| `paper/neurips2026/*` | 新建 | NeurIPS 2026 格式论文（main.tex + 5 sections + checklist） |
| `CLAUDE.md` | 新建 | 项目指引文件 |
