# ScholarGym TODO

## 检索模块 (`rag.py`, `build_vector_db.py`)

- [x] [P0] `search_citations_hybrid()` 调用 `self.search_citations()`(FAISS) 但 `load_or_build_indices()` 加载的是 Qdrant，hybrid 模式会崩溃。统一为 Qdrant 路径
- [x] [P0] BM25 和 vector 的 rank 计算不一致：FAISS `search_citations` rank 从 1-indexed+offset → 统一为 0-indexed（`rag.py:626`）
- [x] [P1] 统一向量后端为 Qdrant，清理 FAISS 相关的构建/加载/查询代码（`build_vector_library`、`load_vector_library`、`search_citations`）
- [x] [P1] 统一 Embedding 路径：`build_vector_db.py` 和 `rag.py` 都用 OllamaEmbeddings，消除 SentenceTransformer 双轨问题（或反之）
- [x] [P1] `build_vector_db.py` 硬编码的 `QDRANT_URL`/`OLLAMA_URL` 改为读 `config.py`
- [ ] [P2] BM25 tokenization (`\b[a-z]+\b`) 会丢弃数字，"BERT-2" 只保留 "bert"（`rag.py:128`）
- [ ] [P3] checkpoint 每 batch 全量写 `indexed_keys` JSON，改为追加写或降低写入频率
- [ ] [P3] `search_citations_vector()` 通过 LangChain 封装层查询，考虑直接用 Qdrant client `search()` API 减少开销
- [ ] [P3] 评估是否需要调整 Qdrant 的 HNSW 索引参数（`m`、`ef_construct`）以优化 570K 规模下的查询速度

## Agent 模块 (`agent/`)

- [x] [P0] `deeprag.py:364` ID 类型不匹配：`selected_paper_ids_tracker` 和 `final_selected_papers` 去重均改为 `p.arxiv_id`
- [x] [P1] `selector.py:47` 返回类型声明为 `Tuple[List[Paper], str]`，实际返回 3 元组
- [x] [P1] `planner.py:217` root query continue 未支持：`existing_ids.add(0)` 被注释，planner 尝试 continue root 时可能 KeyError
- [x] [P2] `planner.py:208` 缺少 early return：complete 且无 subqueries 时应提前退出
- [x] [P2] `planner.py:232` 缺少 link_type 和 text 组合的校验
- [x] [P2] `browser.py:204` `soup.head.title` 在 `soup.head` 为 None 时抛 AttributeError
- [x] [P2] `browser.py:547-584` 测试代码留在生产模块中，应移至 tests/（已随死代码清理删除）
- [ ] [P2] `browser.py:244` 硬编码 30s timeout，无 exponential backoff
- [ ] [P2] `summarizer.py:72-73` 缓存写入非原子操作（文件写 + 内存更新），非线程安全
- [x] [P2] `structures.py:91` `SelectorOutputWithBrowser.to_browse` 声明为 `List[dict]`，实际代码当 `Dict[str, dict]` 用

## 工作流 (`deeprag.py`, `simplerag.py`, `eval.py`)

- [ ] [P1] `deeprag.py:164` 和 `eval.py:145` 缺少 workflow failure / early stop 处理
- [ ] [P1] checkpoint resume 功能标记为 TODO 但未实现（`eval.py:293/314/324/341/426`）
- [ ] [P1] checkpoint rebuild (`utils.py:558`) 缺少 `f1`/`retrieval_f1` 指标，恢复后的 summary 与 fresh run 不一致
- [ ] [P2] `eval.py:616` simple workflow 的输出目录未区分，应使用 top_k 而非 results_per_query

## 代码质量 (`utils.py`, `prompt.py`, `logger.py`)

- [x] [P0] `utils.py:36` 正则贪婪匹配 → 改为括号平衡提取 `_extract_outermost_json()`，避免灾难性回溯
- [ ] [P1] `prompt.py` 模板变量无 XML 转义：`{query}`、paper content 等直接注入 XML 结构的 prompt，畸形摘要可能破坏 prompt 结构
- [ ] [P2] `logger.py:50` `LoggerHandler.log` 列表无界增长，无日志轮转
- [ ] [P3] `utils.py:569` checkpoint rebuild 中 `max()` 可能作用于空 dict

## 异常处理 (`deeprag.py`, `api.py`, `agent/`, `utils.py`)

### Critical — 导致整个 query/workflow 崩溃

- [ ] [P0] `deeprag.py` `run()` 方法零异常保护（line 102-503）：planner/retrieval/selector 任何一步失败，整个 query 崩溃。需在主循环内加 try/except，失败时记录并跳过当前迭代或返回部分结果
- [ ] [P0] `deeprag.py` `asyncio.gather` 未设 `return_exceptions=True`（line 207/245/274/293）：一个并发任务失败导致所有已完成任务结果丢失
- [ ] [P0] `utils.py:503` `CheckpointManager.load_checkpoint` JSON 解析无 try/except：checkpoint 文件损坏（如进程被 kill 写了半行）= 全部进度丢失
- [ ] [P1] Planner/Selector/Summarizer 的 LLM 调用均无异常保护（`planner.py:162`、`selector.py:124`、`summarizer.py:132`）。仅 Browser 有外层 catch 兜底

### High — 资源泄漏或数据风险

- [ ] [P1] `api.py:130` sync OpenAI client 创建后未关闭（async 版本 line 238 有 `client.close()`，sync 版本没有），长时间运行积累未关闭的 HTTP 连接
- [ ] [P1] `api.py` 无 timeout 配置（line 134-141）：远程 API hang 住时调用无限等待
- [x] [P1] `planner.py:228` LLM 输出未验证直接 `int()`：`int(item.get("target_k", ...))` 若 LLM 返回非数字字符串则 ValueError 崩溃
- [x] [P1] `retrieval_mcp.py:24` hybrid 搜索未传 `gt_arxiv_ids`/`exclude_arxiv_ids`，与 bm25/vector 分支接口不一致
- [ ] [P2] `rag.py:196` pickle 加载和 `rag.py:109` JSON 加载均无 try/except，文件损坏或 OOM 直接崩溃

### Medium — 错误被吞或处理不当

- [ ] [P2] `eval.py:442` 过宽的 `except Exception` 吞掉编程错误（TypeError/KeyError 等 bug 被当作"正常失败"跳过）
- [ ] [P2] `api.py:151-153` 结构化输出失败后静默降级为文本调用，如果失败原因是网络错误则不应重试相同请求
- [ ] [P2] `browser.py:259` `except (httpx.TimeoutException, httpx.RequestError, Exception)` 中 Exception 使前两个冗余，分类判断失效
- [ ] [P2] `utils.py:349/352` `process_arxiv_jsonl_file` 错误路径用 `print` 而非 `logger`
- [ ] [P2] `browser.py:506-509` `_execute_llm_call` debug 文件写入无 try/except（对比 `browse_papers:362-367` 有保护）
- [ ] [P2] `summarizer.py:72-79` 缓存写入后无 `f.flush()`，且文件写和内存更新非原子（进程被 kill 时数据不一致）

## 部署 (`docker-compose`, `scripts/`)

- [x] [P2] 编写 `docker-compose.yml`，管理 Qdrant 服务，数据卷持久化
- [ ] [P2] 编写 `scripts/start_services.sh`：启动 docker-compose + Ollama、健康检查、可选自动构建索引

## 数据分发 (HuggingFace)

- [ ] [P1] 导出 Qdrant 快照并上传 HuggingFace：`curl -X POST http://localhost:6433/collections/paper_knowledge_base/snapshots`，快照文件约 3-4 GB
- [ ] [P1] 编写 `scripts/restore_qdrant.py`：下载快照 + 恢复到本地 Qdrant，一条命令完成
- [ ] [P1] 上传预构建的 `bm25_index.pkl`（2 GB）到 HuggingFace，免去用户构建时间
- [ ] [P2] 更新 README Quick Start：补充 docker-compose + 数据恢复步骤
- [ ] [P2] 确认 Ollama 是放进 docker-compose 还是保持本地运行

## 实验 (`eval.py`, `config_*.py`)

- [ ] [P1] MiniMax M2.5 实验：scnet 账户余额不足（402），需充值后启动
- [ ] [P1] GLM-5 实验：zhipu API key 过期（401），需更新 key 后启动
- [ ] [P1] Claude Sonnet 4.6 实验：API 已通，实验未启动
- [ ] [P2] GPT-5.4 全量实验结果收集与分析（300 queries, 已在后台运行）

## 论文 (`paper/neurips2026/`)

- [ ] [P1] 论文表格更新：等新模型实验完成后更新 Table 1-4 数据
- [ ] [P1] 论文写作润色：减少 AI 风格总结语句，使用地道学术英语
- [ ] [P2] LaTeX 编译验证：本地和远程均未安装 LaTeX，NeurIPS 版本未验证是否能编译
- [ ] [P3] 新模型 BibTeX 条目：GPT-5.4、Claude Sonnet 4.6、MiniMax M2.5、GLM-5

## 冗余模块清理

### 死代码 — 可直接删除

- [x] [P1] `code/utils.py` 构建阶段残留 — 已删除 `extract_ground_truth_titles()`、`clean_text_content()`、`parse_bib()`、`parse_authors_parsed()`、`generate_arxiv_url()`、`process_arxiv_entry()`、`process_arxiv_jsonl_file()`、`__main__` 块（约 150 行） — `extract_ground_truth_titles()`（已被 arxiv_id 版本取代）、`clean_text_content()`、`parse_bib()`、`parse_authors_parsed()`、`generate_arxiv_url()`、`process_arxiv_entry()`、`process_arxiv_jsonl_file()` 以及 `__main__` 块（约 150 行），build_bench.py 自带副本，utils 里的已无调用者
- [x] [P2] `code/logger.py:_set_transformers_logging()` — 已删除，移除 `transformers` import
- [x] [P2] `code/agent/browser.py:4-14` — 已删除注释掉的 path-fixing 代码
- [x] [P2] `code/agent/browser.py:547-584` — 已删除内嵌 `test_browser()` 和 `__main__` 块

### 预留接口 — 保留，不删

- `code/mcp/pdf_mcp.py` — MCP 接口 placeholder（获取论文章节/引用/被引），与 browser.py 的 Ar5ivParser 互补，是有意预留的扩展点
- `code/mcp/retrieval_mcp.py:batch_search_papers()` — 批量检索接口，当前逐 subquery 调用但未来可能批量优化
- `code/logger.py:reset_logging()` — jupyter/多实验场景下的工具函数
- `code/rag.py:batch_search_*` — 批量检索接口，预留扩展
- `code/rag.py:get_paper_by_id()` — 合理的 lookup 工具
- `code/fast_build_bench.py` — build_bench 的 HTTP API 替代方案（arxiv 库限流时备用），与 build_bench 重复但功能独立

### 待架构统一时清理 — 随 Step 2 一起处理

- [x] [P1] `code/rag.py` FAISS 路径 — `load_vector_library()`、`search_citations()`、`search_citations_hybrid()`、`merge_citation_files()`、`display_search_results()`、`__main__` 块。依赖已切换到 Qdrant，FAISS 代码不再可达，等 Step 2 Qdrant 收口时统一清理

### 独立脚本 — 移至 `scripts/`

- [ ] [P2] `code/test_qwen.py` → 移至 `tests/`（LLM API 冒烟测试）
- [ ] [P2] `code/estimate_closed_model_metrics.py` → `scripts/`（一次性分析，硬编码绝对路径）
- [ ] [P2] `code/fix_closed_model_metrics.py` → `scripts/`（一次性数据处理）
- [ ] [P2] `code/plot_evaluation_summary.py` → `scripts/`（论文图表生成）
- [ ] [P2] `code/build.py` → `scripts/`（语料爬取）
- [ ] [P2] `code/build_bench.py` → `scripts/`（benchmark 构建）
- [ ] [P2] `code/build_vector_db.py` → `scripts/`（Qdrant 索引构建）

### 配置 / 数据问题

- [x] [P0] `config.py:12` `BENCHMARK_PATH = 'data/scholargym_bench_short.jsonl'` — 已改为 `scholargym_bench.jsonl`
- [x] [P1] `rag.py` `__main__` 引用了 `config.CITED_DATA_DIR` 和 `config.QDRANT_PATH`，但这些字段在任何 config 中均未定义，运行会崩溃（已随 FAISS 清理删除 `__main__` 块）
- [ ] [P2] `data/scholargym_bench_1q.jsonl`、`data/superlong_bench_300.jsonl` — 无任何 config 或代码引用，确认是否仍需保留

## 重构（整体规划，均未执行）

- [ ] [P3] Phase 0: 创建 `scholargym/` 包结构 + Pydantic Settings 配置
- [ ] [P3] Phase 1: 消除重复代码（prompt 参数化、API sync/async 统一、build 去重、Agent 基类）
- [ ] [P3] Phase 2: 拆分 utils.py、分离 GT 与检索、数据模型清理（LinkType enum 等）
- [ ] [P3] Phase 3: 依赖注入、Workflow 基类、deeprag 状态封装
- [ ] [P3] Phase 4: 测试基础设施、错误处理、requirements 版本锁定 + CPU fallback
- [ ] [P3] Phase 5: CLI 提取、包最终化

---

## 建议执行规划（代码架构优先）

以下按依赖关系和风险排序，每个 Step 内的任务可并行。原则：先修正确性 bug，再统一架构，再补健壮性。

### Step 1: 修 P0 — 影响正确性的 bug ✅ 已完成

### Step 2: 检索架构统一 — Qdrant 收口 ✅ 已完成

- FAISS 全部删除，统一为 Qdrant
- hybrid search 删除，只保留 bm25 和 vector
- build_vector_db.py 读 config + CLI args
- docker-compose.yml 创建
- faiss-gpu / sentence-transformers 从 requirements.txt 移除

### Step 3: Agent 模块修正 ✅ 已完成

1. `selector.py:47` 返回类型对齐（改声明或改实现）
2. `planner.py:217` 解开 root continue 注释，补 id=0 的处理逻辑
3. `planner.py:208/232` early return + link_type 校验
4. `structures.py:91` `to_browse` 类型对齐
5. `browser.py:204` 加 `soup.head` None 检查

完成标志：`--workflow deep_research --max_iterations 5` 跑完无 warning/error。

### Step 4: 异常处理加固（~1.5 天）

核心目标：一个 query 失败不能拖垮整个 evaluation。

1. `deeprag.py` `run()` 主循环加 try/except：planner/retrieval/selector 失败时记录 warning、跳过当前迭代或返回已有部分结果
2. `asyncio.gather` 全部加 `return_exceptions=True`，调用方检查返回值中的 Exception 并 graceful 处理
3. `CheckpointManager.load_checkpoint` 加 JSON 解析保护：损坏行跳过，不崩溃
4. Agent LLM 调用加 try/except：Planner 失败返回空 subqueries + is_complete=True；Selector 失败返回原始 papers；Summarizer 失败返回空 dict
5. `api.py` sync client 改用 `with` 上下文管理器或手动 close；加 timeout 配置
6. `planner.py:228` `int()` 加 try/except，回退到 `config.MAX_RESULTS_PER_QUERY`
7. ~~`retrieval_mcp.py:24` hybrid 搜索补齐参数~~ ✅ 已在 Step 2 完成

完成标志：手动 kill Ollama 再跑 eval，单 query 报 warning 但 evaluation 继续执行不崩溃。

### Step 5: 工作流健壮性（~2 天）

1. **Checkpoint resume 实现**（`eval.py` 5 处 TODO）：跑到一半挂了能从断点恢复
2. Checkpoint rebuild 补 `f1`/`retrieval_f1` 指标
3. `deeprag.py` early stop / failure 处理
4. Simple workflow 输出目录区分

完成标志：kill 进程后重跑，能跳过已完成 query 且 summary 指标一致。

### Step 6: 代码质量与工具链（~1 天）

1. `prompt.py` 模板变量加 XML 转义（写个 `escape_xml()` 工具函数）
2. `logger.py` 改用 `RotatingFileHandler` 或 `deque(maxlen=N)`
3. `utils.py:569` 空 dict 保护
4. `eval.py:442` 区分可恢复错误和编程 bug（catch 具体异常类型而非裸 Exception）
5. `utils.py:349/352` print 改 logger
6. `summarizer.py` 缓存写入加 `f.flush()`

### Step 7: 部署一键化（~1 天）

1. ~~编写 `docker-compose.yml`~~ ✅ 已完成
2. 编写 `scripts/start_services.sh`
3. 验证 `docker compose up -d && python code/eval.py --search_method vector` 端到端通过

### 后续（视需要）

- 大规模重构（Phase 0-5）建议在论文投稿后再做，当前阶段改动太大会影响实验可复现性
- BM25 tokenization 改进、Qdrant HNSW 调参等性能优化可按需穿插
