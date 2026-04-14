"""
ScholarGym demo — MOCK mode.

No dependencies on Qdrant / Ollama / the real pipeline. Useful for UI
development and offline demos. The real-pipeline bridge lives in
`server.py` alongside this file.

Run:  python code/workflow_demo/server_mock.py  (port 8766)
"""

import asyncio, json, random, uuid
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))

# ── Mock paper database ───────────────────────────────────────────────

PAPERS = [
    {"arxiv_id":"2401.15884","title":"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection","authors":["Akari Asai","Zeqiu Wu","Yizhong Wang","Avirup Sil","Hannaneh Hajishirzi"],"year":2024,"venue":"ICLR 2024","abstract":"Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection.","citations":487,"url":"https://arxiv.org/abs/2401.15884","relevance":0.95},
    {"arxiv_id":"2312.10997","title":"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval","authors":["Parth Sarthi","Salman Abdullah","Aditi Tuli","Shubh Khanna","Anna Goldie","Christopher D. Manning"],"year":2024,"venue":"ICLR 2024","abstract":"Retrieval-augmented language models can better adapt to changes in world state and incorporate long-tail knowledge. However, most existing methods retrieve only short contiguous chunks from a retrieval corpus, limiting holistic understanding of the overall document context. We introduce the novel approach of recursively embedding, clustering, and summarizing chunks of text, constructing a tree with differing levels of summarization from the bottom up.","citations":312,"url":"https://arxiv.org/abs/2312.10997","relevance":0.91},
    {"arxiv_id":"2305.14283","title":"Active Retrieval Augmented Generation","authors":["Zhengbao Jiang","Frank F. Xu","Luyu Gao","Zhiqing Sun","Qian Liu","Jane Dwivedi-Yu","Yiming Yang","Jamie Callan","Graham Neubig"],"year":2023,"venue":"EMNLP 2023","abstract":"Despite the remarkable ability of large language models (LLMs) to comprehend and generate language, they have a tendency to hallucinate and create factually inaccurate output. Augmenting LLMs by retrieving information from external knowledge resources can partially alleviate this issue. We propose Forward-Looking Active REtrieval augmented generation (FLARE), a generic method that iteratively uses a prediction of the upcoming sentence to anticipate future content, which is then utilized as a query to retrieve relevant documents.","citations":256,"url":"https://arxiv.org/abs/2305.14283","relevance":0.88},
    {"arxiv_id":"2310.08901","title":"Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models","authors":["Wenhao Yu","Hongming Zhang","Xiaoman Pan","Kaixin Ma","Hongwei Wang","Dong Yu"],"year":2023,"venue":"arXiv preprint","abstract":"Retrieval-augmented language models (RALMs) represent an advancing field, aiming to improve large language models by incorporating external knowledge. However, existing RALMs face notable challenges: they tend to falter when retrieved documents are noisy or irrelevant. We introduce Chain-of-Noting (CoN), a novel approach designed to improve the robustness of RALMs in facing noisy, irrelevant documents and in handling unknown scenarios.","citations":178,"url":"https://arxiv.org/abs/2310.08901","relevance":0.85},
    {"arxiv_id":"2402.18679","title":"Corrective Retrieval Augmented Generation","authors":["Shi-Qi Yan","Jia-Chen Gu","Yun Zhu","Zhen-Hua Ling"],"year":2024,"venue":"ICML 2024","abstract":"Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be guaranteed by solely relying on their parametric knowledge. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong. We propose the Corrective Retrieval Augmented Generation (CRAG) to improve the robustness of generation.","citations":203,"url":"https://arxiv.org/abs/2402.18679","relevance":0.82},
    {"arxiv_id":"2401.04088","title":"Lost in the Middle: How Language Models Use Long Contexts","authors":["Nelson F. Liu","Kevin Lin","John Hewitt","Ashwin Paranjape","Michele Bevilacqua","Fabio Petroni","Percy Liang"],"year":2024,"venue":"TACL 2024","abstract":"While recent language models have the ability to take long contexts as input, relatively little is known about how well they use longer context. We analyze the performance of language models across multiple tasks that require identifying relevant information within long input contexts. We find that performance is often highest when relevant information occurs at the beginning or end of the input context.","citations":891,"url":"https://arxiv.org/abs/2401.04088","relevance":0.78},
    {"arxiv_id":"2305.06983","title":"Query Rewriting for Retrieval-Augmented Large Language Models","authors":["Xinbei Ma","Yeyun Gong","Pengcheng He","Hai Zhao","Nan Duan"],"year":2023,"venue":"EMNLP 2023","abstract":"Large Language Models (LLMs) play powerful, black-box readers in the retrieve-then-read pipeline, making remarkable progress in knowledge-intensive tasks. This work focuses on the rewrite-retrieve-read framework rather than the previous retrieve-then-read, introducing a trainable scheme for the rewriting module.","citations":145,"url":"https://arxiv.org/abs/2305.06983","relevance":0.75},
    {"arxiv_id":"2312.05934","title":"Dense X Retrieval: What Retrieval Granularity Should We Use?","authors":["Tong Chen","Hongwei Wang","Sihao Chen","Wenhao Yu","Kaixin Ma","Xinran Zhao","Hongming Zhang","Dong Yu"],"year":2024,"venue":"ACL 2024","abstract":"Dense retrieval has become a prominent method to obtain relevant context or world knowledge in open-domain NLP tasks. When we use a learned dense retriever, an often-overlooked design choice is the retrieval unit in which texts are indexed, e.g., document, passage, or sentence. We systematically study the impact of retrieval granularity and introduce a novel retrieval unit, proposition.","citations":98,"url":"https://arxiv.org/abs/2312.05934","relevance":0.71},
]

SUBQUERY_POOL = [
    [("retrieval augmented generation survey 2024", "derive", 10), ("self-reflective RAG methods", "expand", 8)],
    [("dense retrieval for scientific literature", "continue", 10), ("corrective RAG robustness", "derive", 8), ("query rewriting for RAG", "expand", 6)],
    [("RAG vs fine-tuning knowledge-intensive tasks", "derive", 8)],
]


def _metrics(it, mx):
    base = 0.15 + 0.2 * it
    r = min(max(round(base + random.uniform(-0.05, 0.08), 3), 0), 1)
    p = min(max(round(base * 0.75 + random.uniform(-0.03, 0.06), 3), 0), 1)
    f = round(2*r*p/(r+p), 3) if (r+p) > 0 else 0
    return {"recall": r, "precision": p, "f1": f}


class S:
    def __init__(self, ws): self.ws = ws
    async def send(self, ev):
        await self.ws.send_text(json.dumps(ev, ensure_ascii=False))


async def simulate(ws, config):
    s = S(ws)
    mi = config.get("max_iterations", 3)
    bm = config.get("browser_mode", "NONE")
    sm = config.get("enable_summarization", False)
    query_text = config.get("query", "Recent advances in retrieval-augmented generation for scientific literature")
    gt = ["2401.15884", "2312.10997", "2305.14283", "2310.08901"]

    await s.send({"type": "status", "status": "thinking", "message": "Starting deep research..."})
    await asyncio.sleep(0.3)

    # ── Loading ──
    await s.send({"type": "step", "id": "load", "label": "Initializing environment", "status": "running", "collapsible": True})
    await asyncio.sleep(0.4)
    for t in ["Paper database: 570,412 papers", "BM25 index loaded", "Agents: Planner, Selector, Browser, Summarizer"]:
        await s.send({"type": "step", "id": f"l-{t[:6]}", "parent": "load", "label": t, "status": "done"})
        await asyncio.sleep(0.2)
    await s.send({"type": "step", "id": "load", "label": "Environment ready", "status": "done"})
    await asyncio.sleep(0.2)

    found = set()
    selected_papers = []
    total_sel = 0

    for it in range(mi):
        iid = f"it{it}"
        pool = SUBQUERY_POOL[it % len(SUBQUERY_POOL)]
        await s.send({"type": "step", "id": iid, "label": f"Iteration {it+1}/{mi}", "status": "running", "collapsible": True, "variant": "iteration"})
        await asyncio.sleep(0.15)

        # Planner
        pid = f"{iid}-plan"
        await s.send({"type": "step", "id": pid, "parent": iid, "label": "Planner: planning subqueries...", "status": "running", "variant": "agent"})
        await asyncio.sleep(random.uniform(0.8, 1.5))
        ns = len(pool)
        lines = []
        for i, (t, lt, k) in enumerate(pool):
            pfx = "└─" if i == ns-1 else "├─"
            lines.append(f"  {pfx} [{lt}] \"{t}\" (k={k})")
        is_comp = it == mi - 1 and random.random() < 0.2
        lines.append(f"  is_complete: {is_comp}")
        await s.send({"type": "step", "id": pid, "parent": iid, "label": f"Planner: {ns} subqueries", "status": "done", "variant": "agent", "detail": "\n".join(lines)})
        await asyncio.sleep(0.15)

        if is_comp and total_sel > 0:
            await s.send({"type": "step", "id": f"{iid}-stop", "parent": iid, "label": "Research marked complete — stopping", "status": "done", "variant": "info"})
            await s.send({"type": "step", "id": iid, "label": f"Iteration {it+1}/{mi} (early stop)", "status": "done"})
            break

        # Retrieval
        rid = f"{iid}-ret"
        await s.send({"type": "step", "id": rid, "parent": iid, "label": f"Retrieving papers ({ns} subqueries)...", "status": "running", "variant": "phase", "collapsible": True})
        ret_data = {}
        for si, (sq, lt, k) in enumerate(pool):
            sid = f"{rid}-{si}"
            method = random.choice(["BM25", "Vector", "Hybrid"])
            await s.send({"type": "step", "id": sid, "parent": rid, "label": f"{method}: \"{sq}\"", "status": "running", "variant": "tool"})
            await asyncio.sleep(random.uniform(0.2, 0.6))
            nr = random.randint(5, k+2)
            papers = random.sample(PAPERS, min(nr, len(PAPERS)))
            ret_data[si] = {"papers": papers, "count": nr, "sq": sq}
            await s.send({"type": "step", "id": sid, "parent": rid, "label": f"{method}: {nr} papers", "status": "done", "variant": "tool"})
        tr = sum(d["count"] for d in ret_data.values())
        await s.send({"type": "step", "id": rid, "parent": iid, "label": f"Retrieved {tr} papers total", "status": "done", "variant": "phase"})
        await asyncio.sleep(0.15)

        # Summarizer
        if sm:
            smid = f"{iid}-sum"
            await s.send({"type": "step", "id": smid, "parent": iid, "label": f"Summarizer: compressing {tr} papers...", "status": "running", "variant": "agent"})
            await asyncio.sleep(random.uniform(0.4, 1.0))
            ch = random.randint(0, tr//2)
            await s.send({"type": "step", "id": smid, "parent": iid, "label": f"Summarizer: done ({ch} cached)", "status": "done", "variant": "agent"})
            await asyncio.sleep(0.1)

        # Selector
        selid = f"{iid}-sel"
        await s.send({"type": "step", "id": selid, "parent": iid, "label": f"Selector: filtering...", "status": "running", "variant": "phase", "collapsible": True})
        to_browse = []
        for si, rd in ret_data.items():
            ssid = f"{selid}-{si}"
            await s.send({"type": "step", "id": ssid, "parent": selid, "label": f"Selector: \"{rd['sq']}\"", "status": "running", "variant": "agent"})
            await asyncio.sleep(random.uniform(0.5, 1.2))
            nsel = random.randint(1, min(3, rd["count"]))
            ndisc = rd["count"] - nsel
            nbr = random.randint(0, min(2, ndisc)) if bm != "NONE" else 0
            total_sel += nsel
            picks = random.sample(rd["papers"], min(nsel, len(rd["papers"])))
            for p in picks:
                if p["arxiv_id"] in gt:
                    found.add(p["arxiv_id"])
                if p not in selected_papers:
                    selected_papers.append(p)
            if nbr > 0:
                to_browse.extend(random.sample(rd["papers"], min(nbr, len(rd["papers"]))))
            await s.send({"type": "step", "id": ssid, "parent": selid, "label": f"Selector: +{nsel} sel, -{ndisc} disc" + (f", ?{nbr}" if nbr else ""), "status": "done", "variant": "agent"})
        await s.send({"type": "step", "id": selid, "parent": iid, "label": f"Selector: {total_sel} selected so far", "status": "done", "variant": "phase"})
        await asyncio.sleep(0.1)

        # Browser
        if bm != "NONE" and to_browse:
            bid = f"{iid}-br"
            nb = len(to_browse)
            await s.send({"type": "step", "id": bid, "parent": iid, "label": f"Browser: fetching {nb} full texts...", "status": "running", "variant": "agent"})
            await asyncio.sleep(random.uniform(0.5, 1.2))
            ok = random.randint(max(1, nb-1), nb)
            await s.send({"type": "step", "id": bid, "parent": iid, "label": f"Browser: {ok} fetched ({bm})", "status": "done", "variant": "agent"})
            await asyncio.sleep(0.1)
            rsid = f"{iid}-rsel"
            await s.send({"type": "step", "id": rsid, "parent": iid, "label": f"re-Selector: re-evaluating...", "status": "running", "variant": "agent"})
            await asyncio.sleep(random.uniform(0.3, 0.8))
            promo = random.randint(0, min(2, ok))
            total_sel += promo
            await s.send({"type": "step", "id": rsid, "parent": iid, "label": f"re-Selector: +{promo} promoted", "status": "done", "variant": "agent"})

        # Memory
        await s.send({"type": "step", "id": f"{iid}-mem", "parent": iid, "label": f"Memory updated · {total_sel} papers tracked", "status": "done", "variant": "info"})

        m = _metrics(it, mi)
        await s.send({"type": "metrics", "parent": iid, "metrics": m, "found_gt": len(found), "total_gt": len(gt)})
        await s.send({"type": "step", "id": iid, "label": f"Iteration {it+1}/{mi}", "status": "done", "variant": "iteration", "duration": round(random.uniform(2, 5), 1)})
        await asyncio.sleep(0.2)

    await s.send({"type": "status", "status": "done", "message": f"Found {len(selected_papers)} relevant papers"})

    # ── Send final paper results ──
    # Deduplicate and sort by relevance
    seen = set()
    unique = []
    for p in selected_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    unique.sort(key=lambda x: -x["relevance"])

    await s.send({
        "type": "results",
        "papers": unique[:8],
        "query": query_text,
        "total_found": len(unique),
        "gt_found": len(found),
        "gt_total": len(gt),
    })


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try: msg = json.loads(data)
            except: msg = {"action": data}
            if msg.get("action") == "search":
                await simulate(ws, msg.get("config", {}))
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8766)
