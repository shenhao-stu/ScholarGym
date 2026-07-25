"""Scatter plot: avg retrieved per iter (x) vs retrieval recall (y) on Test-Fast.

Data is parsed from the experiment dashboard HTML (Main results + Extra Table E1).
Run:
    python scripts/plots/plot_recall_vs_retrieved.py \\
        --dashboard-url http://10.176.55.210:8080/ \\
        --out-dir scripts/plots/out
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# ---------- HTML fetch / load ---------------------------------------------------

def load_html(source: str) -> str:
    if source.startswith(("http://", "https://")):
        env = os.environ.copy()
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(k, None)
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        with opener.open(source, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8")


# ---------- Table parsing -------------------------------------------------------

def _table_after_heading(soup: BeautifulSoup, heading_substr: str):
    for h2 in soup.find_all("h2"):
        if heading_substr in h2.get_text():
            wrap = h2.find_next("div", class_="tablewrap")
            if wrap is None:
                continue
            return wrap.find("table", class_="paper")
    raise RuntimeError(f"heading not found: {heading_substr}")


def _model_name(td) -> str:
    return re.sub(r"\s+", " ", td.get_text()).strip()


def parse_main_recall_fast(soup: BeautifulSoup) -> dict[str, tuple[str, float]]:
    """Returns {display_model: (run_name, ret_recall_fast)}.

    Skips the Direct Query Baseline group (those rows use the simple workflow
    and don't have an Extra Table E1 counterpart).
    """
    table = _table_after_heading(soup, "Main results")
    out: dict[str, tuple[str, float]] = {}
    skip = True
    for tr in table.find("tbody").find_all("tr"):
        if "grouprow" in (tr.get("class") or []):
            label = tr.get_text().strip().lower()
            skip = "baseline" in label
            continue
        if "bottomrule" in (tr.get("class") or []):
            continue
        cells = tr.find_all("td")
        if not cells or "model" not in (cells[0].get("class") or []):
            continue
        if skip:
            continue
        model = _model_name(cells[0])
        # Test-Fast columns: R, P, F1, Ret.R, Ret.P, Ret.F1
        ret_r_cell = cells[4]
        run = ret_r_cell.get("data-run") or ""
        try:
            ret_r = float(ret_r_cell.get("data-raw"))
        except (TypeError, ValueError):
            continue
        out[model] = (run, ret_r)
    return out


def parse_extra_e1_retrieved_mean_fast(soup: BeautifulSoup) -> dict[str, float]:
    """Returns {display_model: mean_retrieved_fast}.

    The Mean cell has no data-field attribute — read its text instead.
    """
    table = _table_after_heading(soup, "Extra Table E1")
    out: dict[str, float] = {}
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if not cells or "model" not in (cells[0].get("class") or []):
            continue
        model = _model_name(cells[0])
        # Test-Fast columns: It.1..It.5, Mean (= cells[1..6])
        mean_cell = cells[6]
        try:
            mean_val = float(mean_cell.get_text().strip())
        except ValueError:
            continue
        out[model] = mean_val
    return out


# ---------- Plot ----------------------------------------------------------------

# Family-coherent palette: each base family shares a hue and varies in
# lightness/saturation by capacity (small → light, large → dark).
# Gemma → teal/green family; Qwen → blue family; GLM/GPT/Claude → distinct
# accent colors. think/no-think variants share a color and are split by marker.
BASE_COLORS: dict[str, str] = {
    # Gemma family (teal → deep green)
    "Gemma-4-E2B":       "#7FCDBB",  # teal-light
    "Gemma-4-E4B":       "#41B6A6",  # teal-mid
    "Gemma-4-31B":       "#1B7C76",  # teal-dark
    # Qwen family (light blue → deep blue)
    "Qwen3.5-9B":        "#9ECAE1",  # blue-light
    "Qwen3.6-27B":       "#4292C6",  # blue-mid
    "Qwen3.6-35B-A3B":   "#08519C",  # blue-dark
    # Single-model families (distinct accents)
    "GLM-5.1":           "#DC0000",  # red
    "GPT-5.4":           "#ff7f0e",  # orange
    "Claude-Sonnet-4.6": "#7E6148",  # brown
}


def _draw_axis_break_marks(ax) -> None:
    """Draw // hash marks on the x and y spines near (0,0) to flag a non-zero axis start."""
    h = 0.012     # half-length of each diagonal (axes-fraction coords)
    sep = 0.022   # spacing between the two parallel slashes
    x_at = 0.018  # where on the bottom spine the x-axis break sits
    y_at = 0.04   # where on the left spine the y-axis break sits
    kwargs = dict(
        transform=ax.transAxes, color="k", clip_on=False,
        linewidth=1.1, solid_capstyle="butt", zorder=10,
    )
    # x-axis break (//, lying on bottom spine)
    ax.plot([x_at - h, x_at + h], [-h, +h], **kwargs)
    ax.plot([x_at - h + sep, x_at + h + sep], [-h, +h], **kwargs)
    # y-axis break (//, lying on left spine)
    ax.plot([-h, +h], [y_at - h, y_at + h], **kwargs)
    ax.plot([-h, +h], [y_at - h + sep, y_at + h + sep], **kwargs)


def plot_scatter(rows, out_dir: Path, draw_labels: bool = False) -> None:
    from matplotlib.lines import Line2D  # 局部导入以防顶部漏加

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # 1. 颜色按 base model 名分组
    for r in rows:
        r["is_think"] = "†" in r["model"]
        r["base"] = r["model"].replace("†", "").strip()

    unique_bases = sorted({r["base"] for r in rows})
    fallback = plt.get_cmap("tab10")
    color_map = {
        base: BASE_COLORS.get(base, fallback(i % fallback.N))
        for i, base in enumerate(unique_bases)
    }
    missing = [b for b in unique_bases if b not in BASE_COLORS]
    if missing:
        print(f"WARN: no palette entry for: {missing} — using tab10 fallback")

    # 2. 绘制散点
    for r in rows:
        x = r["retrieved"]
        y = r["ret_recall"]
        label = r["model"]

        color = color_map[r["base"]]
        marker = "^" if r["is_think"] else "o"  # think 用三角形，nothink 用圆形

        ax.scatter(
            x, y, 
            color=color, 
            marker=marker, 
            s=70, 
            edgecolors="black", 
            linewidths=0.5, 
            zorder=3
        )

        # 控制是否绘制重叠的文本标签
        if draw_labels:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=4,
            )

    # 3. 图例：每个模型单独一项；按 base model 分栏，think/no-think 配对相邻
    # 3 列布局：Gemma 一列、Qwen 一列、Others（GLM/GPT/Claude）一列
    legend_order = [
        "Gemma-4-E2B", "Gemma-4-E4B", "Gemma-4-31B",
        "Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.6-35B-A3B",
        "GLM-5.1", "GPT-5.4", "Claude-Sonnet-4.6",
    ]

    def _sort_key(r):
        idx = legend_order.index(r["base"]) if r["base"] in legend_order else len(legend_order)
        return (idx, r["is_think"])  # no-think (False) 排在 think (True) 前

    legend_elements = []
    for r in sorted(rows, key=_sort_key):
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="^" if r["is_think"] else "o",
                color="w",
                markerfacecolor=color_map[r["base"]],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=9,
                label=r["model"],
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        ncols=3,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.4,
    )

    ax.set_xlabel("Avg. Retrieved per iteration (Test-Fast)")
    ax.set_ylabel("Retrieval Recall (Test-Fast, iter 5)")
    ax.set_title("Recall vs. Retrieval Breadth across Models")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)

    # 两个轴都不是从 0 开始，画一组 // 断口标记提示读者
    _draw_axis_break_marks(ax)

    fig.tight_layout()
    pdf = out_dir / "recall_vs_retrieved.pdf"
    png = out_dir / "recall_vs_retrieved.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"saved: {pdf}")
    print(f"saved: {png}")


# ---------- Main ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard-url",
        default="http://10.176.55.210:8080/",
        help="Dashboard URL or local HTML file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "out",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV dump of the merged data.",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Annotate each point with its model name.",
    )
    args = parser.parse_args()

    html = load_html(args.dashboard_url)
    soup = BeautifulSoup(html, "html.parser")

    recall_map = parse_main_recall_fast(soup)
    retrieved_map = parse_extra_e1_retrieved_mean_fast(soup)

    rows = []
    only_in_recall = []
    only_in_retrieved = []
    for model, (run, ret_r) in recall_map.items():
        if model not in retrieved_map:
            only_in_recall.append(model)
            continue
        rows.append(
            {
                "model": model,
                "run": run,
                "ret_recall": ret_r,
                "retrieved": retrieved_map[model],
            }
        )
    for model in retrieved_map:
        if model not in recall_map:
            only_in_retrieved.append(model)

    if only_in_recall:
        print(f"WARN: in Main results but not in E1: {only_in_recall}")
    if only_in_retrieved:
        print(f"WARN: in E1 but not in Main results: {only_in_retrieved}")

    rows.sort(key=lambda r: r["retrieved"])
    print(f"matched {len(rows)} models")
    for r in rows:
        print(f"  {r['model']:<32s} retrieved={r['retrieved']:6.2f}  Ret.R={r['ret_recall']:.3f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as f:
            f.write("model,run,retrieved_mean_fast,ret_recall_fast\n")
            for r in rows:
                f.write(f"{r['model']},{r['run']},{r['retrieved']},{r['ret_recall']}\n")
        print(f"saved: {args.csv}")

    plot_scatter(rows, args.out_dir, draw_labels=args.draw_labels)


if __name__ == "__main__":
    main()
