# Paper Edits — GT Discard Rate 一致性修正

待手动应用的编辑清单（基于 2026/05/07 的 paper/_NIPS26_ScholarGym 检查结果）。

## 背景：定义不一致

GT Discard Rate 在三处出现，**正文 + 表格 + 代码三者一致，附录公式与它们不一致**（差在分母）。

| 来源 | 公式 | 分母语义 |
|---|---|---|
| 正文 `sec/4_Experiments_01.tex` L42–45 | `|(R∩G) \ S| / |R \ S|` | 所有被丢弃的候选 |
| 附录 `sec/appendix.tex` L320–325 | `|(R∩G) − (S∩G)| / |R∩G|` | 检索到的 GT |
| 代码 `code/metrics.py` L130–139 (`discarded_ratio` → 输出键 `gt_discard_rate`) | `len(R∩G − S) / len(R − S)` | 所有被丢弃的候选 |

## 表格数值来源（已核实）

- 表 `tab:assessment_discard`（`sec/4_Experiments_01.tex` L221–253）数据由 `paper/ICML_actual/scripts/build_paper_tables.py::build_table_assessment()` 生成。
- 该脚本 L333–340 直接读取 `discarded_ratio_iter_5`（即 `metrics.py:136` 的输出），并附注释明确：
  > "GT Discard Rate per paper Eq. 4: |(R∩G)\S| / |R\S| → matches our `discarded_ratio_iter_5` (metrics.py:136)."
- 输入文件：`runs_iter25/evaluation_summary_fast.jsonl`，由 `scripts/exp/summary_split_full_denom.py` 从 `runs_iter25/evaluation_summary.jsonl` 拆分得到（key 名 `discarded_ratio_iter_*`）。
- 表中 0.01%–0.31% 的量级与"分母 = 所有被丢弃候选"一致；若按附录定义重算应在 5%–30% 量级。
- 另注：`code/eval.py:188` 还有一个 `missed_gt_ratio`，分母用 `|R|`（不是 `|R\S|`），与 `discarded_ratio` 数值差 ≤0.02%，**未**进入表格，仅出现在 `detailed_results.jsonl` 内。脚本注释已说明这一点。

**结论：表格数值是正确的（按正文公式），只需修附录。**

## 待编辑

### 1. 修改附录公式（必做）

**文件**：`paper/_NIPS26_ScholarGym/sec/appendix.tex`
**位置**：L320–325（"Ground-Truth Discard Rate" 段落）

将
```latex
\paragraph{Ground-Truth Discard Rate.}
This diagnostic measures Relevance Assessment errors---ground-truth papers retrieved but subsequently discarded:
\begin{equation}
\text{GT Discard Rate} = \frac{|(\mathcal{R} \cap \mathcal{G}) - (\mathcal{S} \cap \mathcal{G})|}{|\mathcal{R} \cap \mathcal{G}|}.
\end{equation}
Lower values indicate better retention of relevant papers.
```

改为
```latex
\paragraph{Ground-Truth Discard Rate.}
This diagnostic measures Relevance Assessment errors as the fraction of discarded candidates that are ground-truth:
\begin{equation}
\text{GT Discard Rate} = \frac{|(\mathcal{R} \cap \mathcal{G}) \setminus \mathcal{S}|}{|\mathcal{R} \setminus \mathcal{S}|}.
\end{equation}
Lower values indicate fewer ground-truth papers are mistakenly discarded by the assessment step.
```

### 2. 命名统一（建议）

正文/表头/附录命名混用：
- 正文 L42, 258, 261：`GT Discard` / `GT Discard Rate`
- 表头 L231：`Disc.\%`
- 表 caption L223：`GT Discard Rate (\%)`
- 附录 L320：`Ground-Truth Discard Rate`

建议正文统一用 **"GT Discard Rate"**，表头保留 `Disc.\%`（注明缩写），附录段落标题改为 `\paragraph{GT Discard Rate.}`。

### 3. 清理旧草稿（建议）

`paper/_NIPS26_ScholarGym/sec/4_Experiments.tex` 未被 `main.tex` 引用，但仍引述 1.03%、1.22% 这种与当前表对不上的旧数（应是早期不同 run 的结果）。建议：
- 重命名为 `4_Experiments.tex.bak`，或
- 直接删除（git 已保留历史）。

避免后续协作者误认为是当前版本。

---

# Paper Edits — Single-Pass Retrieval Baseline 描述与实现对照

## 已核对的代码路径

- `code/workflows/simple.py::SimpleWorkflow.run()`
- `code/utils/metrics_helpers.py::combine_search_results()`
- `code/eval.py:408–433`（simple workflow 分支）
- `paper/ICML_actual/scripts/build_paper_tables.py::simple_subset_metrics()`（表 1 baseline 行的实际生成器）
- prompt：`code/prompt/query_generation.py`（`SIMPLE_QUERY_GENERATION_PROMPT` / `COMPLEX_QUERY_GENERATION_PROMPT`）
- 对照对象：`code/workflows/deep_research.py`、`code/prompt/planner.py`

## 一致性表

| 论文（appendix L391 / 正文 L25）说法 | 代码实现 | 一致性 |
|---|---|---|
| "prompted once to emit a set of search keys" | `generate_query_keys` 一次 LLM 调用 | ✅ |
| "each key independently retrieves its top-k" | per-key `top_k = max(EVAL_TOP_K_VALUES) = 20`（`eval.py:411`） | ✅ |
| "deduplicated union ... merged into a single ranking" | `combine_search_results` 去重后按 **max similarity across keys** 排序 | ✅（实现细节未点明） |
| "metrics at top-10 of this ranking" | `simple_subset_metrics(..., top_k=10)`（`build_paper_tables.py:175`）截断到前 10 | ✅ |
| "no iteration, no LLM-based relevance assessment" | 无 selector / browser / iter 循环 | ✅ |
| "reuses the **prompt template** of the full workflow" | simple 用 `SIMPLE/COMPLEX_QUERY_GENERATION_PROMPT`，deep 用 planner DAG prompt（带 experience replay、target_k、checklist、link_type） | ❌ **不一致** |
| "reuses generation parameters" | 共用 `LLM_GEN_PARAMS`、`ENABLE_REASONING`、`IS_LOCAL_LLM` | ✅ |
| "reuses retrieval backend" | 共用 `CitationRAGSystem` | ✅ |
| "same retrieval budget as the first iteration of the full workflow" | simple 最终 \|R\|=\|S\|=10；deep iter-1 \|R\|≈ N_sub × `MAX_RESULTS_PER_QUERY=10`（≈50），再由 selector 过滤为 \|S\| | ⚠️ **措辞不严谨** |

## 待编辑

### 4. 修正 prompt template 表述（必做）

**文件**：`paper/_NIPS26_ScholarGym/sec/appendix.tex` L391
**问题**：声称 "reuses the prompt template ... of the full workflow"，但 simple workflow 实际用的是 `SIMPLE_QUERY_GENERATION_PROMPT`（key list 输出），与 deep_research planner 的 DAG decomposition prompt 不是同一个模板。

将
```
The baseline reuses the prompt template, generation parameters, and retrieval backend of the full workflow, ...
```
改为（建议措辞）
```
The baseline reuses the same backbone, decoding parameters, and retrieval backend as the full workflow; only the query-decomposition prompt is a simplified single-shot variant that lists search keys without DAG structure or memory.
```

### 5. 澄清 "retrieval budget" 表述（必做）

**文件**：同上 L391
**问题**：simple baseline 报告 top-10（即 \|R\|=10），而 deep workflow iter-1 的检索集 \|R\| ≈ 50。两者并非"same retrieval budget"。

将
```
We report metrics at top-10 of this ranking, which corresponds to the same retrieval budget as the first iteration of the full workflow.
```
改为（建议措辞之一）
```
We report metrics at top-10 of this ranking; this output cardinality matches the typical selection budget per iteration in the full workflow, allowing direct comparison of the final paper set produced without LLM-based assessment.
```

### 6. 在附录补全实现细节（建议）

为复现性，建议在 baseline 段补一句具体参数：
> Concretely, each key retrieves up to 20 candidates; the deduplicated union is sorted by the maximum similarity score across keys (a paper retrieved by multiple keys is ranked by its best score) and truncated to the top 10 for evaluation.

对应代码：
- per-key `top_k=20` —— `eval.py:411` `top_k=max(top_k_list)`，默认 `EVAL_TOP_K_VALUES=[5,10,20]`
- max-similarity 排序 —— `code/utils/metrics_helpers.py:38-42`
- top-10 截断 —— `paper/ICML_actual/scripts/build_paper_tables.py:175`

---

# Paper Edits — Iteration Dynamics figure 插入正文

## 背景

`paper/_NIPS26_ScholarGym/fig/iteration_dynamics.pdf` 是一个 3-panel 图（产自 `scripts/plots/plot_iteration_dynamics.py`）：
- (a) Mean retrieved candidates `|R_t|` per iteration（Test-Fast）
- (b) Mean selected candidates `|S_t|` per iteration（Test-Fast）
- (c) Recall trajectories over 25 iterations（左轴累积 recall + 右轴 ΔRecall）

之前未在正文中引用；§4.2 Iteration Dynamics 仅有一个 wrapfigure 用 `extended_iterations.pdf`（仅 panel(c) 的内容）。

## 已应用（2026/05/07）

### 7. 替换 §4.2 wrapfigure 为全宽 iteration_dynamics 图

**文件**：`paper/_NIPS26_ScholarGym/sec/4_Experiments_01.tex`

- L151–156：删除原 `\begin{wrapfigure}[16]{r}{0.5\linewidth} ... extended_iterations.pdf ... \label{fig:extended_iter}`，替换为全宽 `\begin{figure}[t] ... iteration_dynamics.pdf ... \label{fig:iteration_dynamics}`，caption 重写覆盖三个 panel 含义（含 default `T=5` 的虚线说明）。
- L158：引导句改为 `Table~\ref{tab:iteration_metrics} reports per-iteration recall and precision; Figure~\ref{fig:iteration_dynamics} pairs these with the underlying retrieval/selection volumes (panels (a)--(b)) and extends the recall analysis to 25 iterations (panel (c)).`
- L166：Saturation 段把 `Figure~\ref{fig:extended_iter}` 改为 `Figure~\ref{fig:iteration_dynamics}(c)`。

`extended_iterations.pdf` 文件本身保留在 `fig/`，未删除（如不再使用，可在最终清理时一并移除）。

### 8. 旧草稿引用同步（与上方第 3 条合并）

`paper/_NIPS26_ScholarGym/sec/4_Experiments.tex`（未被 `main.tex` 引用）仍含 `\label{fig:extended_iter}` 与多处 `\ref{fig:extended_iter}`。它本就在第 3 条建议清理范围内；如保留则与 `_01` 版形成同名 label 冲突的潜在风险（不会触发，因为编译时只 include 一份）。处理方式同第 3 条：重命名 `.bak` 或删除。

---

## 验证

修改后建议本地编译确认无 LaTeX 错误：
```bash
cd paper/_NIPS26_ScholarGym && pdflatex main.tex
```
