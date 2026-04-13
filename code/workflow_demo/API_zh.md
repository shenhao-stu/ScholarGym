# ScholarGym Workflow Demo — 接口文档

## 概述

本项目是一个基于 **FastAPI** + **WebSocket** 的单页演示应用。用户提交研究问题后，服务端模拟多轮迭代的深度检索流程，并通过 WebSocket 实时推送进度事件至客户端。

- **服务端**: `server.py` (FastAPI + Uvicorn)
- **客户端**: `index.html` (原生 HTML/CSS/JS，由服务端托管)
- **默认地址**: `http://127.0.0.1:8765`

---

## HTTP 接口

### `GET /`

返回 `index.html` 页面。

| 项目 | 值 |
|------|-----|
| 方法 | `GET` |
| 路径 | `/` |
| 响应类型 | `text/html` |

---

## WebSocket 接口

### `ws://{host}/ws`

所有研究交互均通过此持久化 WebSocket 连接进行。

---

## 客户端 → 服务端消息

客户端向服务端发送 JSON 消息，目前仅支持一种操作：

### `search`

启动一次研究模拟。

```json
{
  "action": "search",
  "config": {
    "query": "Recent advances in retrieval-augmented generation",
    "max_iterations": 3,
    "browser_mode": "REFRESH",
    "enable_summarization": false,
    "source": "Local"
  }
}
```

#### 配置字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `string` | `"Recent advances in retrieval-augmented generation for scientific literature"` | 研究问题 |
| `max_iterations` | `int` | `3` | 研究迭代次数（1–10） |
| `browser_mode` | `string` | `"REFRESH"` | 全文抓取行为，详见 [浏览器模式](#浏览器模式) |
| `enable_summarization` | `bool` | `false` | 是否在每轮迭代中启用摘要代理 |
| `source` | `string` | `"Local"` | 数据源选择（仅前端使用，服务端暂未消费） |

#### 浏览器模式

| 值 | 说明 |
|-----|------|
| `NONE` | 仅检索，不抓取全文。跳过 Browser 代理 |
| `PRE_ENRICH` | 在筛选阶段之前预先抓取全文 |
| `REFRESH` | 初次筛选后，对边界论文抓取全文并重新评估 |

---

## 服务端 → 客户端事件

服务端通过 WebSocket 流式推送 JSON 事件，每个事件包含 `type` 字段。共有四种事件类型：

### 1. `status` — 状态事件

标识整体研究流程的开始或结束。

```json
{
  "type": "status",
  "status": "thinking",
  "message": "Starting deep research..."
}
```

```json
{
  "type": "status",
  "status": "done",
  "message": "Found 6 relevant papers"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `"thinking"` \| `"done"` | 当前阶段 |
| `message` | `string` | 可读的状态描述 |

---

### 2. `step` — 步骤事件

表示研究流程中的单个步骤。步骤支持嵌套（通过 `parent`）和折叠。

```json
{
  "type": "step",
  "id": "it0-plan",
  "parent": "it0",
  "label": "Planner: 2 subqueries",
  "status": "running",
  "variant": "agent",
  "collapsible": true,
  "detail": "  ├─ [derive] \"retrieval augmented generation survey 2024\" (k=10)\n  └─ [expand] \"self-reflective RAG methods\" (k=8)\n  is_complete: false",
  "duration": 3.2
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `string` | 是 | 步骤唯一标识。发送相同 `id` 会更新已有步骤 |
| `parent` | `string` | 否 | 父步骤的 `id`（用于嵌套） |
| `label` | `string` | 是 | 步骤显示文本 |
| `status` | `"running"` \| `"done"` | 是 | 步骤当前状态 |
| `variant` | `string` | 否 | 视觉样式提示，可选值: `iteration`、`agent`、`phase`、`tool`、`info` |
| `collapsible` | `bool` | 否 | 是否可在 UI 中折叠 |
| `detail` | `string` | 否 | 多行详情文本（折叠体内显示） |
| `duration` | `number` | 否 | 耗时（秒），用于迭代完成时展示 |

#### 步骤层级结构（典型）

```
Iteration 迭代 (variant: iteration, 可折叠)
├── Planner 规划器 (variant: agent)
├── Retrieval 检索 (variant: phase, 可折叠)
│   ├── BM25/Vector/Hybrid 子查询 (variant: tool)
│   └── ...
├── Summarizer 摘要器 (variant: agent) — 仅当 enable_summarization=true
├── Selector 筛选器 (variant: phase, 可折叠)
│   ├── 各子查询的筛选 (variant: agent)
│   └── ...
├── Browser 浏览器 (variant: agent) — 仅当 browser_mode ≠ NONE
├── re-Selector 重筛选 (variant: agent) — 仅当浏览器获取了论文
└── Memory 记忆更新 (variant: info)
```

#### 步骤更新模式

步骤按 `id` 进行 **upsert**（插入或更新）。步骤首先以 `status: "running"` 发送，随后更新为 `status: "done"`，`label` 可能会改变。例如：

1. `{ id: "it0-plan", label: "Planner: planning subqueries...", status: "running" }`
2. `{ id: "it0-plan", label: "Planner: 2 subqueries", status: "done", detail: "..." }`

---

### 3. `metrics` — 指标事件

在每轮迭代结束时报告评估指标。

```json
{
  "type": "metrics",
  "parent": "it0",
  "metrics": {
    "recall": 0.32,
    "precision": 0.21,
    "f1": 0.25
  },
  "found_gt": 2,
  "total_gt": 4
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `parent` | `string` | 所属迭代步骤的 `id` |
| `metrics.recall` | `float` | 召回率（0–1） |
| `metrics.precision` | `float` | 精确率（0–1） |
| `metrics.f1` | `float` | F1 分数（0–1） |
| `found_gt` | `int` | 目前已发现的标准答案论文数 |
| `total_gt` | `int` | 标准答案论文总数（本 demo 中固定为 4） |

---

### 4. `results` — 结果事件

包含最终论文结果的事件，在所有迭代完成后发送一次。

```json
{
  "type": "results",
  "query": "Recent advances in retrieval-augmented generation",
  "total_found": 6,
  "gt_found": 3,
  "gt_total": 4,
  "papers": [
    {
      "arxiv_id": "2401.15884",
      "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
      "authors": ["Akari Asai", "Zeqiu Wu", "..."],
      "year": 2024,
      "venue": "ICLR 2024",
      "abstract": "Despite their remarkable capabilities...",
      "citations": 487,
      "url": "https://arxiv.org/abs/2401.15884",
      "relevance": 0.95
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | `string` | 原始研究问题 |
| `total_found` | `int` | 筛选出的唯一论文总数 |
| `gt_found` | `int` | 已找到的标准答案论文数 |
| `gt_total` | `int` | 标准答案论文总数 |
| `papers` | `array` | 最多 8 篇论文，按相关性降序排列。详见 [论文对象](#论文对象) |

#### 论文对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `arxiv_id` | `string` | ArXiv 论文 ID |
| `title` | `string` | 论文标题 |
| `authors` | `string[]` | 作者列表 |
| `year` | `int` | 发表年份 |
| `venue` | `string` | 会议或期刊 |
| `abstract` | `string` | 论文摘要 |
| `citations` | `int` | 引用数 |
| `url` | `string` | ArXiv 链接 |
| `relevance` | `float` | 相关性分数（0–1） |

---

## 事件时序图

```
客户端                                服务端
  │                                    │
  │──── { action: "search", config } ──→│
  │                                    │
  │◄── status (thinking) ─────────────│
  │◄── step: load (running) ──────────│
  │◄── step: load 子步骤 (done) ──────│
  │◄── step: load (done) ─────────────│
  │                                    │
  │  ┌─── 迭代 1 ─────────────────────│
  │◄─│ step: it0 (running)            │
  │◄─│ step: it0-plan (running→done)  │
  │◄─│ step: it0-ret (running→done)   │
  │◄─│ step: it0-sel (running→done)   │
  │◄─│ step: it0-br (running→done)    │  ← 仅当 browser_mode ≠ NONE
  │◄─│ step: it0-mem (done)           │
  │◄─│ metrics                         │
  │◄─│ step: it0 (done)               │
  │  └────────────────────────────────│
  │                                    │
  │  ... 重复每轮迭代 ...              │
  │                                    │
  │◄── status (done) ─────────────────│
  │◄── results ────────────────────────│
  │                                    │
```

---

## 启动方式

```bash
pip install fastapi uvicorn
python server.py
# → 打开 http://127.0.0.1:8765
```
