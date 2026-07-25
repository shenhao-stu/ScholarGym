"""2-panel iteration dynamics figure for the paper.

Panels (left → right):
  A. Bar chart: avg #retrieved / iter (Test-Fast Mean from dashboard Extra
     Table E1). Qwen variants are excluded from this panel.
  B. Dual-axis line: cumulative recall + ΔRecall over up to 25 iterations
     (data from runs_iter25/evaluation_summary_fast.jsonl — Test-Fast subset,
     200 queries — mirroring paper/ICML/scripts/plot_experiments_actual.py::
     plot_extended_iterations).

Run:
    python scripts/plots/plot_iteration_dynamics.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from matplotlib.lines import Line2D


# ---------- Palette (kept in sync with plot_experiments.py::ITER_COLORS) -----

# Custom palette (user-specified): three cool teals/blues for the Gemma family
# and three warm orange/sand hues for the proprietary backbones. Bar order
# matches the order of the listed hex codes.
BASE_COLORS: dict[str, str] = {
    "Gemma-4-E2B":       "#257D8B",  # deep teal
    "Gemma-4-E4B":       "#68BED9",  # light blue
    "Gemma-4-31B":       "#BFDFD2",  # pale mint
    "GLM-5.1":           "#EAA558",  # orange
    "GPT-5.4":           "#ED8D5A",  # coral
    "Claude-Sonnet-4.6": "#EFCE87",  # sandy yellow
    # Qwen variants kept for other plots that may import this constant; not
    # used in the current iteration_dynamics figure (Qwen rows are filtered).
    "Qwen3.5-9B":        "#F39B7F",
    "Qwen3.6-27B":       "#8491B4",
    "Qwen3.6-35B-A3B":   "#91D1C2",
}

# Bar-chart order: Qwen variants and think variants excluded (no-think only).
DISPLAY_ORDER_BARS = [
    "Gemma-4-E2B",
    "Gemma-4-E4B",
    "Gemma-4-31B",
    "GLM-5.1",
    "GPT-5.4",
    "Claude-Sonnet-4.6",
]

# Line-chart order (panel b). No-think only, matches the bar chart.
DISPLAY_ORDER_LINES = list(DISPLAY_ORDER_BARS)

# Map display label → run_name in runs_iter25/evaluation_summary_fast.jsonl
ITER25_RUNS: dict[str, str] = {
    "Gemma-4-E2B":       "gemma4-e2b-nothink-none-iter25",
    "Gemma-4-E2B†":      "gemma4-e2b-think-none-iter25",
    "Gemma-4-E4B":       "gemma4-e4b-nothink-none-iter25",
    "Gemma-4-E4B†":      "gemma4-e4b-think-none-iter25",
    "Gemma-4-31B":       "gemma4-31b-nothink-none-iter25",
    "Gemma-4-31B†":      "gemma4-31b-think-none-iter25",
    "GLM-5.1":           "glm51-nothink-none-iter25",
    "GLM-5.1†":          "glm51-think-none-iter25",
    "GPT-5.4":           "gpt54-nothink-none-iter25",
    "GPT-5.4†":          "gpt54-think-none-iter25",
    "Claude-Sonnet-4.6": "claude-sonnet46-nothink-none-iter25",
}


def _color_for(label: str) -> str:
    base = label.replace("†", "").strip()
    return BASE_COLORS.get(base, "#888888")


def _is_think(label: str) -> bool:
    return "†" in label


# ---------- HTML fetch -------------------------------------------------------

def load_html(source: str) -> str:
    if source.startswith(("http://", "https://")):
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(source, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8")


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


def parse_mean_fast(soup: BeautifulSoup, heading_substr: str) -> dict[str, float]:
    """Both E1 (Retrieved) and E2 (Selected) share the layout: cells[6] is the
    Test-Fast Mean column."""
    table = _table_after_heading(soup, heading_substr)
    out: dict[str, float] = {}
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if not cells or "model" not in (cells[0].get("class") or []):
            continue
        model = _model_name(cells[0])
        try:
            mean_val = float(cells[6].get_text().strip())
        except ValueError:
            continue
        out[model] = mean_val
    return out


# ---------- iter25 trajectory loader ----------------------------------------

def load_iter25_recall(jsonl_path: Path) -> dict[str, list[float]]:
    """Returns {run_name: cumulative_recall_per_iter (forward-filled)}."""
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    out: dict[str, list[float]] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("EVAL_DETAILED_RESULTS_PATH", "")
            run_name = p.split("/")[1] if "runs_iter25/" in p else None
            if not run_name or "shard" in run_name:
                continue
            traj_raw = [r.get(f"recall_iter_{k}") for k in range(1, 26)]
            last = None
            traj = []
            for v in traj_raw:
                if v is not None:
                    last = v
                traj.append(last if last is not None else 0.0)
            out[run_name] = traj  # last write wins (same key would just be a re-eval)
    return out


# ---------- Plot helpers ----------------------------------------------------

def _draw_bars(ax, mean_map: dict[str, float], labels: list[str], ylabel: str) -> None:
    values = [mean_map.get(lab, np.nan) for lab in labels]
    colors = [_color_for(lab) for lab in labels]
    hatches = ["//" if _is_think(lab) else "" for lab in labels]

    x = np.arange(len(labels))
    bars = ax.bar(
        x, values, color=colors, edgecolor="none", zorder=3,
    )
    # apply hatches per-bar (set after creation to keep per-bar control)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Bars sit at zorder=3; lift the bottom spine above them so the axis line
    # is not visually clipped by the bar fills (bottom edge of each bar sits
    # exactly on y=0).
    ax.spines["bottom"].set_zorder(4)

    # value annotations on top of bars
    for xi, v in zip(x, values):
        if np.isnan(v):
            continue
        ax.text(xi, v + max(values) * 0.012, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, zorder=4)


def _draw_extended_iterations(
    ax, traj_map: dict[str, list[float]],
) -> tuple[list[Line2D], list[str]]:
    """Cumulative recall + ΔRecall (secondary axis), NPG palette.

    Same base color is shared between no-think and think variants of a model;
    no-think uses a solid line, think uses dashed.
    """
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ordered = [lbl for lbl in DISPLAY_ORDER_LINES if lbl in ITER25_RUNS]

    main_lines: list[Line2D] = []
    model_labels: list[str] = []
    for label in ordered:
        traj = traj_map.get(ITER25_RUNS[label])
        if traj is None:
            continue
        color = _color_for(label)
        ls_recall = "--" if _is_think(label) else "-"
        ls_delta  = (0, (1, 1)) if _is_think(label) else (0, (4, 2))  # dotted vs dashed
        x = np.arange(1, 26)
        y = np.asarray(traj, dtype=float)
        deltas = np.diff(y)

        ln, = ax.plot(x, y, color=color, linewidth=2.0,
                      linestyle=ls_recall, label=label, zorder=3)
        ax2.plot(x[1:], deltas, color=color, linewidth=1.2,
                 linestyle=ls_delta, alpha=0.7, zorder=2)
        main_lines.append(ln)
        model_labels.append(label)

    ax.axvline(x=5, color="gray", linestyle=":", linewidth=1.5, alpha=0.8, zorder=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Recall")
    ax2.set_ylabel(r"$\Delta$ Recall", color="#333333")
    ax2.tick_params(axis="y", labelcolor="#333333")
    ax.grid(axis="y", linestyle="-", alpha=0.5)

    return main_lines, model_labels


# ---------- Main -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard-url", default="http://10.176.55.210:8080/")
    parser.add_argument(
        "--iter25-jsonl",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "runs_iter25/evaluation_summary_fast.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument(
        "--paper-fig-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper/_NIPS26_ScholarGym/fig",
        help="Also drop a copy of the PDF here (set empty string to skip).",
    )
    args = parser.parse_args()

    # --- data ---
    soup = BeautifulSoup(load_html(args.dashboard_url), "html.parser")
    retrieved_mean = parse_mean_fast(soup, "Extra Table E1")
    traj_map = load_iter25_recall(args.iter25_jsonl)

    print(f"E1 retrieved (Test-Fast): {len(retrieved_mean)} models")
    print(f"iter25 trajectories      : {len(traj_map)} runs")

    # --- figure ---
    plt.rcParams.update({
        # Match NeurIPS 2026 body text (neurips_2026.sty: \rmdefault=ptm =>
        # Times Roman). Use STIX for math so \Delta etc. render in Times-style
        # glyphs instead of the default Computer Modern.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.26)
    ax_ret = fig.add_subplot(gs[0, 0])
    ax_ext = fig.add_subplot(gs[0, 1])

    _draw_bars(ax_ret, retrieved_mean, DISPLAY_ORDER_BARS, "Avg. #Retrieved / iter")
    ax_ret.set_title("(a) Retrieved per iteration (Test-Fast)")

    main_lines, model_labels = _draw_extended_iterations(ax_ext, traj_map)
    ax_ext.set_title("(b) Recall over 25 iterations")

    # Combined legend: per-model lines (color = base, linestyle = think mode)
    # plus a small style key explaining the line semantics.
    style_key = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-"),
        Line2D([0], [0], color="black", lw=1.2, linestyle=(0, (4, 2)), alpha=0.7),
    ]
    style_labels = ["Recall", r"$\Delta$ Recall"]
    model_legend = ax_ext.legend(
        main_lines + style_key,
        model_labels + style_labels,
        loc="center left", bbox_to_anchor=(1.13, 0.5),
        frameon=True, fancybox=True, edgecolor="0.8",
        fontsize=10, ncol=1, labelspacing=0.45, borderpad=0.6,
        handlelength=2.4,
    )

    # tight_layout doesn't account for the off-axis legend in panel (b);
    # leave explicit margins and let bbox_inches='tight' (set in rcParams)
    # expand the saved PDF/PNG to include the legend.
    fig.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.24, wspace=0.30)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / "iteration_dynamics.pdf"
    png = args.out_dir / "iteration_dynamics.png"
    fig.savefig(pdf, bbox_extra_artists=(model_legend,))
    fig.savefig(png, dpi=200, bbox_extra_artists=(model_legend,))
    plt.close(fig)
    print(f"saved: {pdf}")
    print(f"saved: {png}")

    if str(args.paper_fig_dir):
        args.paper_fig_dir.mkdir(parents=True, exist_ok=True)
        target = args.paper_fig_dir / "iteration_dynamics.pdf"
        target.write_bytes(pdf.read_bytes())
        print(f"copied: {target}")


if __name__ == "__main__":
    main()
