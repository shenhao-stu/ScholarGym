# ScholarGym 测试说明

## 概述

本测试套件围绕 eval 工作流细节、指标计算和 DeepResearch 工作流编排进行验证，覆盖核心计算逻辑，不依赖 LLM 调用或真实 RAG 系统。

- **框架**: pytest
- **测试文件**: `test_eval_and_metrics.py`, `test_deeprag_workflow.py`
- **公共配置**: `conftest.py` (sys.path 设置、mock config 注入、共享 fixtures、fixture 加载器)
- **测试数据**: `fixtures/` 目录下的 JSON 文件（与测试逻辑解耦）
- **总计**: 120 个测试

## 目录结构

```
code/tests/
├── conftest.py                     # 公共配置、fixture 加载器、factory fixtures
├── test_eval_and_metrics.py        # eval 与 metrics 模块测试 (70 个测试)
├── test_deeprag_workflow.py        # DeepResearch 工作流测试 (43 个测试)
├── README.md
└── fixtures/
    ├── benchmark_samples.json      # benchmark 数据条目、GT ID 集合
    ├── metrics_cases.json          # subquery/iteration/retrieval/avg_distance 测试用例
    ├── utils_cases.json            # 工具函数、checkpoint 测试用例
    └── deeprag_cases.json          # MCP 转换、selector 处理、workflow 编排测试用例
```

## 运行方式

```bash
conda activate verl
# 运行全部测试
python -m pytest code/tests/ -v

# 单独运行某个文件
python -m pytest code/tests/test_eval_and_metrics.py -v
python -m pytest code/tests/test_deeprag_workflow.py -v

# 运行特定测试类
python -m pytest code/tests/test_eval_and_metrics.py::TestCalculateSubqueryMetrics -v

# 运行特定参数化用例（按 fixture 中的 id 匹配）
python -m pytest code/tests/test_eval_and_metrics.py -k "perfect_match" -v
```

## 测试数据管理

测试数据与测试逻辑完全解耦，存放在 `fixtures/` 目录下的 JSON 文件中。

### 添加新测试用例

只需编辑对应的 JSON fixture 文件，无需修改测试代码。以 `metrics_cases.json` 为例：

```json
{
  "subquery_metrics": [
    {
      "id": "my_new_case",
      "retrieved": ["gt1", "noise1"],
      "selected": ["gt1"],
      "gt": ["gt1", "gt2"],
      "expected": {
        "ret_recall": 0.5, "ret_prec": 0.5,
        "sel_recall": 0.5, "sel_prec": 1.0,
        "disc_count": 0
      }
    }
  ]
}
```

每个用例必须包含 `"id"` 字段，用作 pytest 参数化的标识符（显示在 `-v` 输出中）。

### Fixture 文件说明

| 文件 | 内容 |
|------|------|
| `benchmark_samples.json` | benchmark 条目（5 组）、GT ID 集合（4 组：set_a/set_b/single/large） |
| `metrics_cases.json` | subquery_metrics（10 组）、iteration_metrics（8 组）、retrieval_metrics（8 组）、simple_avg_distance（6 组） |
| `utils_cases.json` | extract_ground_truth（8 组）、parse_json_from_tag（11 组）、remove_think_blocks（8 组）、checkpoint_rebuild（4 组） |
| `deeprag_cases.json` | mcp_results_to_papers（6 组）、process_selector_results（5+3 组）、browser_mode_cases（7 组）、workflow_run 用 papers/subqueries |

### conftest.py 提供的 fixtures

| Fixture | 类型 | 说明 |
|---------|------|------|
| `sample_benchmark_data` | list[dict] | 全部 benchmark 条目（5 组） |
| `gt_ids_a` | set | 3 个 GT ID |
| `gt_ids_b` | set | 5 个 GT ID |
| `gt_ids_single` | set | 1 个 GT ID |
| `gt_ids_large` | set | 10 个 GT ID |
| `metrics_fixtures` | dict | metrics_cases.json 全量数据 |
| `utils_fixtures` | dict | utils_cases.json 全量数据 |
| `deeprag_fixtures` | dict | deeprag_cases.json 全量数据 |
| `make_benchmark_entry` | factory | 按参数生成 benchmark 条目 |
| `make_paper` | factory | 按参数生成 Paper 对象（自增 ID） |

## 测试覆盖范围

### 文件一：`test_eval_and_metrics.py`

#### 1. MetricsCalculator（`code/metrics.py`）

**TestCalculateSubqueryMetrics** — 单子查询指标计算（10 个参数化用例）
- 完美匹配、部分匹配、无重叠
- 空检索/空 GT/全空集合的边界情况
- 被丢弃 GT 论文的追踪（retrieved 但未 selected）
- 单论文完美、大噪声小 GT、全 GT 检索但无选择

**TestCalculateIterationMetrics** — 迭代级指标聚合（8 个参数化用例）
- 多子查询去重后的 Recall/Precision
- F1 计算正确性
- `discarded_ratio` = 被丢弃 GT / 总丢弃数
- 三子查询无重叠、全噪声无 GT 命中
- 空子查询结果字典的边界处理

**TestCalculateGtRankAndDistance** — GT 排名与距离计算（9 个测试）
- 单子查询排名 → 距离公式 `max(1 - rank/cutoff, 0)`
- 多子查询取最小排名
- 历史最佳排名追踪（跨迭代保留最优）
- 超出 cutoff 的距离钳制为 0
- GT 论文完全未找到的情况
- 单 GT rank=1、大 GT 集合部分命中

**TestCalculateSimpleAvgDistance** — 简单工作流平均距离（6 个参数化用例）
- 多个 rank_dict 跨子查询取最小排名后平均
- 空 rank_dicts / 空 gt_ids 返回 0.0
- 全部 rank=1、单 GT 单 rank

#### 2. 统一检索指标（`code/utils.py`）

**TestCalculateRetrievalMetrics** — calculate_retrieval_metrics 函数（8 个参数化用例）
- 仅检索模式（无 selected 参数）
- 检索 + 选择模式
- F1 计算正确性
- 完美检索与选择、大检索小选择
- 空集合边界

#### 3. 工具函数（`code/utils.py`）

**TestExtractGroundTruthArxivIds** — GT arxiv ID 提取（8 个参数化用例）
- 正常提取（label=1 的论文）
- 过滤缺失 arxiv_id 的论文
- 全正/全负 label、单论文、重复 arxiv_id

**TestParseJsonFromTag** — JSON 解析（11 个参数化用例）
- XML 标签提取：`<tag>{...}</tag>`
- 代码块提取：` ```json{...}``` `
- 原始 JSON 对象解析
- 嵌套 JSON、数组值、unicode、boolean/null
- 无效 JSON / 空响应返回 None

**TestRemoveThinkBlocks** — 思考块移除（8 个参数化用例）
- `<thought>...</thought>` 移除
- 多个块、无块、空字符串
- 多行思考块、尾部思考块、仅思考块、非 thought 标签保留

#### 4. CheckpointManager（`code/utils.py`）

**TestCheckpointManager** — 断点续传管理（9 个测试）
- `append_result` + `is_processed` 往返验证
- `load_checkpoint` 从 JSONL 文件加载
- `rebuild_statistics` deep_research 路径（2 个参数化用例）：指标累积、轮次填充
- `rebuild_statistics` simple 路径（2 个参数化用例）：recall@k / precision@k 重算、多 k 值
- 多次 append、加载顺序保持

#### 5. CitationEvaluator（`code/eval.py`）

**TestEvaluateSingleQueryDeepResearch** — 单查询 deep research 评估（5 个测试）
- 跨迭代累积 retrieved/selected 集合正确性
- workflow 返回 None 的异常处理
- `total_discarded_gt_count` 跨迭代累加
- 单 GT 论文场景
- 三迭代渐进 recall 提升（1/3 → 2/3 → 1.0）

**TestEvaluateBenchmark** — 基准评测聚合（3 个测试）
- deep_research 路径：`avg_*` 指标计算、耗时统计
- simple 路径：recall@k / precision@k 累积与平均
- 轮次填充：查询提前停止时，后续迭代继承最后一轮指标值

---

### 文件二：`test_deeprag_workflow.py`

#### 6. 数据转换（`code/deeprag.py`）

**TestMcpResultsToPapers** — MCP 结果转 Paper 对象（6 个参数化用例）
- 正常字段映射（paper_id, title, abstract, arxiv_id, date, score）
- 缺失字段使用默认值（"", "N/A", None）
- 空输入返回空列表
- 单字段、多论文批量转换

#### 7. Selector 结果处理（`code/deeprag.py`）

**TestProcessSelectorResults** — `_process_selector_results` 静态方法（10 个测试）
- **Overwrite 模式**（5 个参数化用例）：覆盖、替换、多子查询、None 容错、缺失 state
- **INCREMENTAL 模式**（3 个参数化用例）：合并去重、空已有、全重复
- Browsing 任务收集（to_browse dict 转移到 papers_for_browsing）
- 空 browsing 不修改 papers_for_browsing

#### 8. 工作流编排（`code/deeprag.py`）

**TestDeepResearchWorkflowRun** — `run()` 主流程，BROWSER_MODE=NONE（12 个测试）
- 单迭代返回结构验证（history, selected_papers, executed_queries）
- **Planner 提前终止**：is_complete=True + 无已选论文 → 返回 None
- **Planner 正常完成**：is_complete=True + 有论文 → 提前返回
- **无子查询**：返回 None
- **跨迭代论文累积**：多次迭代选出的论文合并到最终列表
- **GT 指标**：history 中包含 iteration_metrics, avg_distance, gt_rank
- **计时字段**：planner/retrieval/selector/browser/overhead/total_during
- **Memory 更新**：plan stages 包含正确的 subquery DAG
- **executed_queries**：所有子查询文本被追踪
- **最终去重**：同一论文多次被选也只出现一次
- **空 GT 集合**：avg_distance 为 0.0
- **三迭代运行**：验证 3 plan + 3 select stages

#### 9. BROWSER_MODE 编排（`code/deeprag.py`）

**TestBrowserModeOrchestration** — `run()` 四种浏览模式的数据流（7 个参数化用例）

| 模式 | 数据流 | 测试用例 |
|------|--------|----------|
| **NONE** | selector 一次决策，不浏览 | `none_no_browsing` |
| **PRE_ENRICH** | 先浏览全部论文 → selector 一次决策 | `pre_enrich_browse_before_selector` |
| **REFRESH** | selector → 浏览不确定论文 → 用**全部原始论文**重跑 selector（overwrite） | `refresh_browse_then_reselector_all`, `refresh_no_uncertain_no_rebrowse` |
| **INCREMENTAL** | selector → 浏览不确定论文 → 仅用**浏览过的论文**再次 selector（merge 去重） | `incremental_browse_then_merge`, `incremental_dedup_on_merge`, `incremental_no_uncertain_no_rebrowse` |

每个用例验证：
- 浏览是否被调用 & 浏览论文数量
- selector 调用次数
- 第二次 selector 收到的论文数量（REFRESH=全量 vs INCREMENTAL=仅浏览过的）
- 最终选中论文 ID 列表
- 最终结果无重复论文
