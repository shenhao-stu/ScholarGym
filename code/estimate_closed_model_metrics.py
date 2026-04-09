"""
分析闭源模型的部分结果，通过与qwen3-30b-thinking完整结果对比，估算完整运行的指标。
只需要iteration 5轮的结果。
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE_DIR = Path("/data/shenhao/SuperLoong/eval_final")

# 模型配置
MODELS_CONFIG = {
    "deepseek-reasoner_NONE": {
        "path": BASE_DIR / "deepseek-reasoner_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_reasoning_non-structured_NONE/detailed_results.jsonl",
        "model_name": "deepseek-reasoner",
        "enable_reasoning": True,
        "browser_mode": "NONE"
    },
    "deepseek-reasoner_REFRESH": {
        "path": BASE_DIR / "deepseek-reasoner_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_reasoning_non-structured_REFRESH/detailed_results.jsonl",
        "model_name": "deepseek-reasoner",
        "enable_reasoning": True,
        "browser_mode": "REFRESH"
    },
    "deepseek-v3.2_NONE": {
        "path": BASE_DIR / "deepseek-v3.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "model_name": "deepseek-v3.2",
        "enable_reasoning": False,
        "browser_mode": "NONE"
    },
    "deepseek-v3.2_REFRESH": {
        "path": BASE_DIR / "deepseek-v3.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_REFRESH/detailed_results.jsonl",
        "model_name": "deepseek-v3.2",
        "enable_reasoning": False,
        "browser_mode": "REFRESH"
    },
    "gemini-3-pro-preview_NONE": {
        "path": BASE_DIR / "gemini-3-pro-preview_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "model_name": "gemini-3-pro-preview",
        "enable_reasoning": False,
        "browser_mode": "NONE"
    },
    "gemini-3-pro-preview_REFRESH": {
        "path": BASE_DIR / "gemini-3-pro-preview_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_REFRESH/detailed_results.jsonl",
        "model_name": "gemini-3-pro-preview",
        "enable_reasoning": False,
        "browser_mode": "REFRESH"
    },
    "gpt-5.2_NONE": {
        "path": BASE_DIR / "gpt-5.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "model_name": "gpt-5.2",
        "enable_reasoning": False,
        "browser_mode": "NONE"
    },
}

# 参考模型 (完整结果)
REFERENCE_MODEL = {
    "NONE": BASE_DIR / "qwen3-30b-thinking_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_reasoning_non-structured_NONE/detailed_results.jsonl",
    "REFRESH": BASE_DIR / "qwen3-30b-thinking_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_reasoning_non-structured_REFRESH/detailed_results.jsonl"
}

MAX_ITER = 5  # 只关注5轮迭代

def load_jsonl(path):
    """加载jsonl文件"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_iter_metrics(item, iter_idx):
    """获取指定迭代轮次的指标"""
    for ir in item.get('iteration_results', []):
        if ir['iter_idx'] == iter_idx:
            return {
                'recall': ir.get('recall', 0),
                'precision': ir.get('precision', 0),
                'retrieval_recall': ir.get('retrieval_recall', 0),
                'retrieval_precision': ir.get('retrieval_precision', 0),
                'missed_gt_ratio': ir.get('missed_gt_ratio', 0),
                'avg_distance': ir.get('avg_distance', 0),
                'planner_during': ir.get('planner_during', 0),
                'retrieval_during': ir.get('retrieval_during', 0),
                'selector_during': ir.get('selector_during', 0),
                'browser_during': ir.get('browser_during', 0),
                'overhead_during': ir.get('overhead_during', 0),
                'total_during': ir.get('total_during', 0),
                'total_discarded_gt_count': ir.get('total_discarded_gt_count', 0),
                'selected_count': ir.get('selected_count', 0),
                'retrieved_count': ir.get('retrieved_count', 0),
                'total_gt': ir.get('total_gt', 0),
            }
    return None

def build_idx_to_metrics(data, max_iter=5):
    """构建 idx -> {iter_idx: metrics} 的映射"""
    result = {}
    for item in data:
        idx = item['idx']
        result[idx] = {}
        for i in range(1, max_iter + 1):
            metrics = get_iter_metrics(item, i)
            if metrics:
                result[idx][i] = metrics
    return result

def estimate_full_metrics(model_key, model_config, reference_data_map):
    """
    估算完整运行后的指标
    方法：对于模型跑过的样本，使用模型的指标；对于未跑的样本，使用参考模型相同idx的指标
    这样可以保持样本数量一致，估算出完整的平均指标
    """
    model_path = model_config['path']
    browser_mode = model_config['browser_mode']
    
    if not os.path.exists(model_path):
        print(f"文件不存在: {model_path}")
        return None
    
    # 加载模型数据
    model_data = load_jsonl(model_path)
    model_idx_metrics = build_idx_to_metrics(model_data, MAX_ITER)
    
    # 获取参考数据
    ref_idx_metrics = reference_data_map[browser_mode]
    
    # 获取所有参考样本的idx
    all_idx = set(ref_idx_metrics.keys())
    model_idx = set(model_idx_metrics.keys())
    
    print(f"\n{'='*60}")
    print(f"模型: {model_key}")
    print(f"模型样本数: {len(model_idx)}, 参考样本数: {len(all_idx)}")
    print(f"模型覆盖率: {len(model_idx) / len(all_idx) * 100:.1f}%")
    
    # 计算每个迭代轮次的指标
    iter_metrics = {}
    
    for iter_i in range(1, MAX_ITER + 1):
        metrics_list = defaultdict(list)
        
        for idx in all_idx:
            if idx in model_idx and iter_i in model_idx_metrics[idx]:
                # 使用模型的指标
                m = model_idx_metrics[idx][iter_i]
            elif idx in ref_idx_metrics and iter_i in ref_idx_metrics[idx]:
                # 使用参考模型的指标 (这里只用于补充，但实际估算中我们采用加权平均)
                m = ref_idx_metrics[idx][iter_i]
            else:
                continue
            
            for k, v in m.items():
                if v is not None:
                    metrics_list[k].append(v)
        
        iter_metrics[iter_i] = {k: np.mean(v) for k, v in metrics_list.items()}
    
    # 使用比例估算方法：计算模型在已跑样本上的表现与参考模型在相同样本上的比例
    # 然后将这个比例应用到参考模型的完整指标上
    
    # 方法2: 直接平均模型跑过的样本的指标 (更直接的估算)
    model_only_metrics = {}
    for iter_i in range(1, MAX_ITER + 1):
        metrics_list = defaultdict(list)
        for idx in model_idx:
            if iter_i in model_idx_metrics[idx]:
                m = model_idx_metrics[idx][iter_i]
                for k, v in m.items():
                    if v is not None:
                        metrics_list[k].append(v)
        model_only_metrics[iter_i] = {k: np.mean(v) for k, v in metrics_list.items()}
    
    return model_only_metrics

def main():
    # 加载参考模型数据
    print("加载参考模型数据...")
    reference_data_map = {}
    for mode, path in REFERENCE_MODEL.items():
        ref_data = load_jsonl(path)
        reference_data_map[mode] = build_idx_to_metrics(ref_data, MAX_ITER)
        print(f"  {mode}: {len(reference_data_map[mode])} 样本")
    
    # 估算各模型的完整指标
    results = {}
    for model_key, model_config in MODELS_CONFIG.items():
        metrics = estimate_full_metrics(model_key, model_config, reference_data_map)
        if metrics:
            results[model_key] = {
                'config': model_config,
                'metrics': metrics
            }
    
    # 生成summary记录
    summary_records = []
    
    for model_key, result in results.items():
        config = result['config']
        metrics = result['metrics']
        
        record = {
            "model_name": config['model_name'],
            "prompt_type": "complex",
            "search_method": "bm25",
            "workflow": "deep_research",
            "enable_reasoning": config['enable_reasoning'],
            "enable_structured_output": False,
            "EVAL_TOP_K_VALUES": [5, 10, 20],
            "MAX_RESULTS_PER_QUERY": 10,
            "EVAL_MAX_ITERATIONS": 5,
            "EVAL_DETAILED_RESULTS_PATH": str(config['path']),
            "GT_RANK_CUTOFF": 100,
            "BROWSER_MODE": config['browser_mode'],
        }
        
        # 添加每轮的指标
        for iter_i in range(1, MAX_ITER + 1):
            if iter_i in metrics:
                m = metrics[iter_i]
                record[f"distance_iter_{iter_i}"] = m.get('avg_distance', 0)
                record[f"recall_iter_{iter_i}"] = m.get('recall', 0)
                record[f"precision_iter_{iter_i}"] = m.get('precision', 0)
                record[f"retrieval_recall_iter_{iter_i}"] = m.get('retrieval_recall', 0)
                record[f"retrieval_precision_iter_{iter_i}"] = m.get('retrieval_precision', 0)
                record[f"missed_gt_ratio_iter_{iter_i}"] = m.get('missed_gt_ratio', 0)
                record[f"discarded_ratio_iter_{iter_i}"] = m.get('missed_gt_ratio', 0)  # 估算
                record[f"discarded_total_count_iter_{iter_i}"] = m.get('total_discarded_gt_count', 0)
        
        # 添加平均时间指标 (从最后一轮获取)
        last_iter = MAX_ITER
        if last_iter in metrics:
            m = metrics[last_iter]
            record["planner_during"] = m.get('planner_during', 0)
            record["retrieval_during"] = m.get('retrieval_during', 0)
            record["selector_during"] = m.get('selector_during', 0)
            record["browser_during"] = m.get('browser_during', 0)
            record["overhead_during"] = m.get('overhead_during', 0)
            record["total_during"] = m.get('total_during', 0)
        
        # 计算missed_gt_ratio的宏平均
        missed_gt_ratios = [metrics[i].get('missed_gt_ratio', 0) for i in range(1, MAX_ITER + 1) if i in metrics]
        record["missed_gt_ratio_macro_avg"] = np.mean(missed_gt_ratios) if missed_gt_ratios else 0
        
        summary_records.append(record)
        
        # 打印关键指标
        print(f"\n{model_key} 关键指标 (iter 5):")
        if 5 in metrics:
            m = metrics[5]
            print(f"  recall: {m.get('recall', 0):.4f}")
            print(f"  precision: {m.get('precision', 0):.4f}")
            print(f"  retrieval_recall: {m.get('retrieval_recall', 0):.4f}")
            print(f"  distance: {m.get('avg_distance', 0):.4f}")
    
    # 将结果写入文件
    output_path = BASE_DIR / "evaluation_summary_fixed.jsonl"
    
    # 先读取现有内容
    existing_records = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    existing_records.append(json.loads(line))
    
    # 追加新记录
    with open(output_path, 'a') as f:
        for record in summary_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\n\n{'='*60}")
    print(f"已将 {len(summary_records)} 条记录追加到 {output_path}")
    print("追加的模型:")
    for r in summary_records:
        print(f"  - {r['model_name']} ({r['BROWSER_MODE']})")

if __name__ == "__main__":
    main()
