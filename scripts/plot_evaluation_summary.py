import json
import os
import sys
import matplotlib.pyplot as plt
from itertools import cycle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import config  # noqa: E402

CUTOFF = 100

def load_jsonl(file_path):
    """Load data from a JSONL (JSON Lines) file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, save_path):
    """Save data list to a JSONL file."""
    with open(save_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"[INFO] Saved completed JSONL to {save_path}")

def ensure_dir(path):
    """Ensure the output directory exists (create if not)."""
    os.makedirs(path, exist_ok=True)

def shorten_model_name(name):
    """Optionally shorten or format model names for display."""
    return name

def assign_model_colors(data_dict):
    """Assign distinct colors to each (model_name, reasoning_flag, structured_flag) triple."""
    triples = sorted({
        (model_name, reasoning_flag, structured_flag)
        for (_, model_name, _, reasoning_flag, structured_flag) in data_dict.keys()
    })
    color_cycle = cycle(plt.cm.tab20.colors)
    return {triple: next(color_cycle) for triple in triples}

def get_linestyle(k):
    """Return line style based on parameter k."""
    k = int(k)
    if k == 20:
        return '-'
    elif k == 10:
        return '--'
    else:
        return ':'

def extract_family(model_name):
    """Extract model family prefix."""
    return model_name.split(':', 1)[0] if ':' in model_name else model_name

def assign_family_markers(data_dict):
    """Assign a unique marker style for each model family."""
    families = sorted({family for (family, *_) in data_dict.keys()})
    marker_cycle = cycle(['o', 's', 'D', 'x', '^', 'v', '*', 'p', 'h'])
    return {family: next(marker_cycle) for family in families}

def plot_line(data_dict, title, xlabel, ylabel, save_path, ylim=None):
    """Plot iteration-based metrics."""
    plt.figure(figsize=(15, 5))
    
    color_map = assign_model_colors(data_dict)
    marker_map = assign_family_markers(data_dict)

    for (family, model_name, k, reasoning_flag, structured_flag), values in data_dict.items():
        color_key = (model_name, reasoning_flag, structured_flag)

        label = f"{shorten_model_name(model_name)}, maxQ={k}, {reasoning_flag}, {structured_flag}"
        linestyle = get_linestyle(k)
        color = color_map[color_key]
        marker = marker_map[family]
        
        plt.plot(
            range(1, len(values) + 1),
            values,
            marker=marker,
            label=label,
            linestyle=linestyle,
            color=color
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.grid(True, linestyle='--', alpha=0.6)
    max_iter = max(len(v) for v in data_dict.values())
    plt.xticks(range(1, max_iter + 1))
    
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_runtime(runtime_dict, save_path):
    """Plot runtime comparison across different processing stages."""
    plt.figure(figsize=(15, 5))
    stages = ["Planner", "Retrieval", "Selector"]
    x = range(len(stages))

    color_map = assign_model_colors(runtime_dict)
    marker_map = assign_family_markers(runtime_dict)

    for (family, model_name, k, reasoning_flag, structured_flag), runtimes in runtime_dict.items():
        color_key = (model_name, reasoning_flag, structured_flag)

        label = f"{shorten_model_name(model_name)}, maxQ={k}, {reasoning_flag}, {structured_flag}"
        linestyle = get_linestyle(k)
        color = color_map[color_key]
        marker = marker_map[family]

        plt.plot(
            x, runtimes,
            marker=marker,
            label=label,
            linestyle=linestyle,
            color=color
        )

    plt.xticks(x, stages)
    plt.title("Runtime by Stage")
    plt.xlabel("Stage")
    plt.ylabel("Duration (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_single_retrieval_precision(retrieval_precision_data, save_path):
    """单独绘制 retrieval_precision_iter 图，并自动调节纵轴范围"""
    plt.figure(figsize=(15, 5))
    
    # 自动分配颜色和标记
    color_map = assign_model_colors(retrieval_precision_data)
    marker_map = assign_family_markers(retrieval_precision_data)
    
    # 计算纵轴范围（自动缩放）
    all_vals = [v for values in retrieval_precision_data.values() for v in values if v is not None]
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.05
        ylim = (max(0, ymin - margin), min(1, ymax + margin))
    else:
        ylim = (0, 1)
    
    for (family, model_name, k, reasoning_flag, structured_flag), values in retrieval_precision_data.items():
        color_key = (model_name, reasoning_flag, structured_flag)
        label = f"{shorten_model_name(model_name)}, maxQ={k}, {reasoning_flag}, {structured_flag}"
        linestyle = get_linestyle(k)
        color = color_map[color_key]
        marker = marker_map[family]

        plt.plot(
            range(1, len(values) + 1),
            values,
            marker=marker,
            label=label,
            linestyle=linestyle,
            color=color
        )

    plt.title("Retrieval Precision vs Iteration (Auto Scaled)")
    plt.xlabel("Iteration")
    plt.ylabel("Retrieval Precision")
    plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.6)

    max_iter = max(len(v) for v in retrieval_precision_data.values())
    plt.xticks(range(1, max_iter + 1))

    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f"[INFO] Saved optimized retrieval precision figure to {save_path}")

def plot_avg_retrieved_per_iter(data, output_dir):
    """绘制平均每轮检索到的文章数量"""
    avg_retrieved_dict = {}

    for entry in data:
        jsonl_path = entry.get("EVAL_DETAILED_RESULTS_PATH")
        if not jsonl_path or not os.path.exists(jsonl_path):
            continue

        model_name = entry["model_name"]
        family = extract_family(model_name)
        k = entry["MAX_RESULTS_PER_QUERY"]
        reasoning_flag = "reasoning" if entry.get("enable_reasoning", False) else "non_reasoning"
        structured_flag = "structured" if entry.get("enable_structured_output", False) else "unstructured"
        key = (family, model_name, k, reasoning_flag, structured_flag)

        per_iter_counts = []
        total_iters = 0

        # 读取每个样本
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                iters = d.get("iteration_results", [])
                for i, it in enumerate(iters, 1):
                    retrieved = it.get("current_iter_retrieved", [])
                    if len(per_iter_counts) < i:
                        per_iter_counts.append([])
                    per_iter_counts[i - 1].append(len(retrieved))
                total_iters = max(total_iters, len(iters))

        # 平均化每轮检索数
        avg_counts = [sum(c) / len(c) if c else 0 for c in per_iter_counts]
        avg_retrieved_dict[key] = avg_counts

    # 绘图
    if not avg_retrieved_dict:
        print("[WARN] No retrieved data found to plot average retrieved per iteration.")
        return

    save_path = os.path.join(output_dir, "avg_retrieved_per_iteration.png")
    plot_line(
        avg_retrieved_dict,
        "Average Retrieved Papers per Iteration",
        "Iteration",
        "Avg Retrieved Papers",
        save_path
    )


def plot_avg_selected_per_iter(data, output_dir):
    """绘制平均每轮保留的文章数量"""
    avg_selected_dict = {}

    for entry in data:
        jsonl_path = entry.get("EVAL_DETAILED_RESULTS_PATH")
        if not jsonl_path or not os.path.exists(jsonl_path):
            continue

        model_name = entry["model_name"]
        family = extract_family(model_name)
        k = entry["MAX_RESULTS_PER_QUERY"]
        reasoning_flag = "reasoning" if entry.get("enable_reasoning", False) else "non_reasoning"
        structured_flag = "structured" if entry.get("enable_structured_output", False) else "unstructured"
        key = (family, model_name, k, reasoning_flag, structured_flag)

        per_iter_counts = []
        total_iters = 0

        # 读取每个样本
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                iters = d.get("iteration_results", [])
                for i, it in enumerate(iters, 1):
                    retrieved = it.get("current_iter_selected", [])
                    if len(per_iter_counts) < i:
                        per_iter_counts.append([])
                    per_iter_counts[i - 1].append(len(retrieved))
                total_iters = max(total_iters, len(iters))

        # 平均化每轮检索数
        avg_counts = [sum(c) / len(c) if c else 0 for c in per_iter_counts]
        avg_selected_dict[key] = avg_counts

    # 绘图
    if not avg_selected_dict:
        print("[WARN] No retrieved data found to plot average retrieved per iteration.")
        return

    save_path = os.path.join(output_dir, "avg_selected_per_iteration.png")
    plot_line(
        avg_selected_dict,
        "Average Selected Papers per Iteration",
        "Iteration",
        "Avg Selected Papers",
        save_path
    )

# ====== 重新计算 ======
def calculate_average_distance(it_info, cutoff, original_cutoff, selected_min_rank_tracker, selected_paper_ids_tracker, gt_arxiv_ids):
    cur_iter_min_rank = {}
    # Step 1: Calculate minimum rank within current iteration across all subqueries
    for q_ranks in it_info.get("gt_rank", []):
        for rank_info in q_ranks.get("ranks", []):
            arxiv_id = rank_info.get("arxiv_id")
            # 需要考虑一下 total_rank 为 -1 和 rank 为 cutoff 的情况，此时置为 config.TOTAL_PAPER_NUM
            if rank_info.get("total_rank") == -1 or rank_info.get("rank") == original_cutoff:
                rank = config.TOTAL_PAPER_NUM
            else:
                rank = rank_info.get("rank")
            # 维护当前轮次的最小 rank
            if arxiv_id not in cur_iter_min_rank:
                cur_iter_min_rank[arxiv_id] = rank
            else:
                cur_iter_min_rank[arxiv_id] = min(cur_iter_min_rank[arxiv_id], rank)
                    
    # Step 2: Apply historical best ranks for selected papers
    adjusted_min_rank = cur_iter_min_rank.copy()
    for arxiv_id in selected_paper_ids_tracker:
        if arxiv_id in (gt_arxiv_ids or []) and arxiv_id in selected_min_rank_tracker:
            if arxiv_id in adjusted_min_rank:
                # Use the better of current rank or historical rank
                adjusted_min_rank[arxiv_id] = min(
                    adjusted_min_rank[arxiv_id],
                    selected_min_rank_tracker[arxiv_id]
                )
            else:
                # Paper not found in current iteration, use historical rank
                adjusted_min_rank[arxiv_id] = selected_min_rank_tracker[arxiv_id]
    
    # Step 3: Update selected_min_rank_tracker for next iteration
    updated_selected_min_rank_tracker = selected_min_rank_tracker.copy()
    for arxiv_id in (gt_arxiv_ids or []):
        if arxiv_id in adjusted_min_rank:
            updated_selected_min_rank_tracker[arxiv_id] = min(
                updated_selected_min_rank_tracker.get(arxiv_id, float('inf')),
                adjusted_min_rank[arxiv_id]
            )
    
    # Step 4: Calculate distance metrics based on adjusted ranks
    cur_iter_distances = {
        arxiv_id: max(1 - (rank / cutoff), 0)
        for arxiv_id, rank in adjusted_min_rank.items()
    }
    
    # Step 5: Calculate average distance
    avg_distance = sum(cur_iter_distances.values()) / len(cur_iter_distances)
    
    return {
        "cur_iter_distances": cur_iter_distances,
        "avg_distance": avg_distance,
        "updated_selected_min_rank_tracker": updated_selected_min_rank_tracker
    }
        
        
                
                
def recompute_metric(metric_name, entry):
    """
    Recompute metric by loading JSONL and averaging over samples.
    E.g., metric_name = "recall_iter_2" or "retrieval_precision_iter_1"
    """
    jsonl_path = entry.get("EVAL_DETAILED_RESULTS_PATH", "")
    if not jsonl_path or not os.path.exists(jsonl_path):
        print(f"[WARN] JSONL path not found for {metric_name}: {jsonl_path}")
        return None

    try:
        base, iter_str = metric_name.rsplit("_iter_", 1)
        iter_idx = int(iter_str)
        metric = base
        if metric == "distance":
            metric = "avg_distance"
    except Exception as e:
        print(f"[ERROR] Invalid metric name: {metric_name} ({e})")
        return None

    total, acc = 0, 0.0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            gt_arxiv_ids = data.get("gt_arxiv_ids", [])
            selected_min_rank_tracker = {}
            selected_paper_ids_tracker = set()
            iters = data.get("iteration_results", [])
            for it in iters:
                cur_iter = it.get("iter_idx", it.get("iteration"))
                selected_paper_ids_tracker.update(it.get("current_iter_selected", []))
                if cur_iter == iter_idx:
                    if metric == "avg_distance" and entry.get("GT_RANK_CUTOFF", 500) != CUTOFF:
                        rank_distance_result = calculate_average_distance(it, CUTOFF, entry.get("GT_RANK_CUTOFF", 500), selected_min_rank_tracker, selected_paper_ids_tracker, gt_arxiv_ids)
                        acc += rank_distance_result["avg_distance"]
                        selected_min_rank_tracker = rank_distance_result["updated_selected_min_rank_tracker"]
                        total += 1
                    elif metric in it:
                        acc += it[metric]
                        total += 1
                    elif metric == "precision":
                        tp = it.get("matches", 0)
                        denom = it.get("selected_count", 0)
                        if denom > 0:
                            acc += tp / denom
                            total += 1
                    elif metric == "retrieval_precision":
                        tp = it.get("retrieval_matches", 0)
                        denom = it.get("retrieved_count", 0)
                        if denom > 0:
                            acc += tp / denom
                            total += 1
                    break

    if total == 0:
        print(f"[WARN] No samples found for {metric_name}")
        return None

    avg_val = acc / total
    # print(f"[INFO] recomputed {metric_name}: {avg_val:.4f} (n={total})")
    return avg_val

# ===============================

def main():
    file_path = "eval_all/evaluation_summary_deduped.jsonl"
    output_dir = "./figure"
    ensure_dir(output_dir)

    data = load_jsonl(file_path)
    completed_records = []  # 完整记录

    # Containers
    distance_data = {}
    recall_data = {}
    precision_data = {}
    retrieval_recall_data = {}
    retrieval_precision_data = {}
    runtime_data = {}
    discarded_ratio_data = {}
    discarded_count_data = {}

    for entry in data:
        model_name = entry["model_name"]
        family = extract_family(model_name)
        k = entry["MAX_RESULTS_PER_QUERY"]
        reasoning_flag = "reasoning" if entry.get("enable_reasoning", False) else "non_reasoning"
        structured_flag = "structured" if entry.get("enable_structured_output", False) else "unstructured"
        key = (family, model_name, k, reasoning_flag, structured_flag)
        # iters = entry["EVAL_MAX_ITERATIONS"]
        iters = 5

        # 若缺失则重新计算
        def safe_extract(prefix):
            vals = []
            for i in range(1, iters + 1):
                key_name = f"{prefix}_{i}"
                if key_name in entry and (
                    prefix != "distance_iter" or entry.get("GT_RANK_CUTOFF", 100) == CUTOFF
                ):
                    val = entry[key_name]
                else:
                    val = recompute_metric(key_name, entry)
                    if val is not None:
                        entry[key_name] = val  # 写回 entry
                vals.append(val)
            return vals

        distances = safe_extract("distance_iter")
        recalls = safe_extract("recall_iter")
        precisions = safe_extract("precision_iter")
        retrieval_recalls = safe_extract("retrieval_recall_iter")
        retrieval_precisions = safe_extract("retrieval_precision_iter")

        discarded_ratios = [entry.get(f"discarded_ratio_iter_{i}", 0) for i in range(1, iters + 1)]
        discarded_counts = [entry.get(f"discarded_total_count_iter_{i}", 0) for i in range(1, iters + 1)]
        runtimes = [
            entry.get("planner_during", 0),
            entry.get("retrieval_during", 0),
            entry.get("selector_during", 0),
        ]

        distance_data[key] = distances
        recall_data[key] = recalls
        precision_data[key] = precisions
        retrieval_recall_data[key] = retrieval_recalls
        retrieval_precision_data[key] = retrieval_precisions
        runtime_data[key] = runtimes
        discarded_ratio_data[key] = discarded_ratios
        discarded_count_data[key] = discarded_counts

        completed_records.append(entry)  # 保存补全后的条目

    # ==== 保存补全后的 JSONL ====
    jsonl_dir = os.path.dirname(file_path)
    completed_path = f"{jsonl_dir}/evaluation_summary_completed_cutoff-{CUTOFF}.jsonl"
    ensure_dir(os.path.dirname(completed_path))
    save_jsonl(completed_records, completed_path)

    # ==== 绘图 ====
    recall_min = min(min(v) for v in recall_data.values())
    recall_max = max(max(v) for v in recall_data.values())
    retr_recall_min = min(min(v) for v in retrieval_recall_data.values())
    retr_recall_max = max(max(v) for v in retrieval_recall_data.values())
    recall_ylim = (min(recall_min, retr_recall_min), max(recall_max, retr_recall_max))

    precision_min = min(min(v) for v in precision_data.values())
    precision_max = max(max(v) for v in precision_data.values())
    retr_precision_min = min(min(v) for v in retrieval_precision_data.values())
    retr_precision_max = max(max(v) for v in retrieval_precision_data.values())
    precision_ylim = (min(precision_min, retr_precision_min), max(precision_max, retr_precision_max))

    plot_line(distance_data, "Effect of Iteration on Distance",
              "Iteration", "Distance",
              os.path.join(output_dir, "distance_vs_iteration.png"))

    plot_line(recall_data, "Effect of Iteration on Recall",
              "Iteration", "Recall",
              os.path.join(output_dir, "recall_vs_iteration.png"),
              ylim=recall_ylim)

    plot_line(precision_data, "Effect of Iteration on Precision",
              "Iteration", "Precision",
              os.path.join(output_dir, "precision_vs_iteration.png"),
              ylim=precision_ylim)

    plot_line(retrieval_recall_data, "Retrieval Recall over Iteration",
              "Iteration", "Retrieval Recall",
              os.path.join(output_dir, "retrieval_recall_vs_iteration.png"),
              ylim=recall_ylim)

    plot_line(retrieval_precision_data, "Retrieval Precision over Iteration",
              "Iteration", "Retrieval Precision",
              os.path.join(output_dir, "retrieval_precision_vs_iteration.png"),
              ylim=precision_ylim)

    plot_line(discarded_ratio_data, "Effect of Iteration on Discarded Ratio",
              "Iteration", "Discarded Ratio",
              os.path.join(output_dir, "discarded_ratio_vs_iteration.png"))

    plot_line(discarded_count_data, "Effect of Iteration on Discarded Count",
              "Iteration", "Discarded Count",
              os.path.join(output_dir, "discarded_count_vs_iteration.png"))

    plot_runtime(runtime_data, os.path.join(output_dir, "runtime_by_stage.png"))
    plot_avg_retrieved_per_iter(data, output_dir)
    plot_avg_selected_per_iter(data, output_dir)
    # ==== 单独绘制 retrieval_precision_iter 自适应比例图 ====
    optimized_fig_path = os.path.join(output_dir, "retrieval_precision_vs_iteration_autoscale.png")
    plot_single_retrieval_precision(retrieval_precision_data, optimized_fig_path)


if __name__ == "__main__":
    main()
