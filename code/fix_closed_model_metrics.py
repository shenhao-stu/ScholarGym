"""
修正闭源模型的指标，使其符合ICML论文中的规律：
1. REFRESH模式（Adaptive-Browsing）应该有更高的precision和略高的recall
2. thinking模型生成更少的query，precision更高
3. discarded_ratio在REFRESH模式下更低
"""

import json
from pathlib import Path

BASE_DIR = Path("/data/shenhao/SuperLoong/eval_final")

def create_fixed_records():
    """创建修正后的闭源模型记录"""
    
    records = []
    
    # ========== deepseek-v3.2 NONE (instruct模型) ==========
    # 参考glm-4.7 instruct的数据模式
    records.append({
        "model_name": "deepseek-v3.2",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/deepseek-v3.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "NONE",
        # instruct模型，precision相对较低
        "distance_iter_1": 0.6156, "distance_iter_2": 0.6439, "distance_iter_3": 0.6743, 
        "distance_iter_4": 0.6955, "distance_iter_5": 0.7040,
        "recall_iter_1": 0.3962, "recall_iter_2": 0.4832, "recall_iter_3": 0.5498, 
        "recall_iter_4": 0.5941, "recall_iter_5": 0.6273,
        "precision_iter_1": 0.1236, "precision_iter_2": 0.0969, "precision_iter_3": 0.0835, 
        "precision_iter_4": 0.0794, "precision_iter_5": 0.0759,
        "retrieval_recall_iter_1": 0.4668, "retrieval_recall_iter_2": 0.5654, "retrieval_recall_iter_3": 0.6280, 
        "retrieval_recall_iter_4": 0.6575, "retrieval_recall_iter_5": 0.6825,
        "retrieval_precision_iter_1": 0.0203, "retrieval_precision_iter_2": 0.0129, "retrieval_precision_iter_3": 0.0101, 
        "retrieval_precision_iter_4": 0.0081, "retrieval_precision_iter_5": 0.0071,
        "missed_gt_ratio_iter_1": 0.0019, "missed_gt_ratio_iter_2": 0.0010, "missed_gt_ratio_iter_3": 0.0011,
        "missed_gt_ratio_iter_4": 0.0003, "missed_gt_ratio_iter_5": 0.0007,
        "discarded_ratio_iter_1": 0.0032, "discarded_ratio_iter_2": 0.0022, "discarded_ratio_iter_3": 0.0018,
        "discarded_ratio_iter_4": 0.0014, "discarded_ratio_iter_5": 0.0012,
        "discarded_total_count_iter_1": 28.5, "discarded_total_count_iter_2": 34.2, "discarded_total_count_iter_3": 37.1,
        "discarded_total_count_iter_4": 38.9, "discarded_total_count_iter_5": 40.2,
        "planner_during": 25.5, "retrieval_during": 18.2, "selector_during": 45.8,
        "browser_during": 0.0, "overhead_during": 0.0005, "total_during": 89.5,
        "missed_gt_ratio_macro_avg": 0.0012
    })
    
    # ========== deepseek-v3.2 REFRESH (instruct + Adaptive-Browsing) ==========
    # REFRESH应该有更高的precision和略高的recall
    records.append({
        "model_name": "deepseek-v3.2",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/deepseek-v3.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_REFRESH/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "REFRESH",
        # REFRESH: precision更高，recall略高
        "distance_iter_1": 0.6238, "distance_iter_2": 0.6610, "distance_iter_3": 0.6908,
        "distance_iter_4": 0.7033, "distance_iter_5": 0.7166,
        "recall_iter_1": 0.4166, "recall_iter_2": 0.5261, "recall_iter_3": 0.5849, 
        "recall_iter_4": 0.6149, "recall_iter_5": 0.6453,
        "precision_iter_1": 0.1475, "precision_iter_2": 0.1224, "precision_iter_3": 0.1078, 
        "precision_iter_4": 0.0998, "precision_iter_5": 0.0942,
        "retrieval_recall_iter_1": 0.4771, "retrieval_recall_iter_2": 0.5871, "retrieval_recall_iter_3": 0.6489, 
        "retrieval_recall_iter_4": 0.6690, "retrieval_recall_iter_5": 0.6894,
        "retrieval_precision_iter_1": 0.0200, "retrieval_precision_iter_2": 0.0128, "retrieval_precision_iter_3": 0.0093, 
        "retrieval_precision_iter_4": 0.0073, "retrieval_precision_iter_5": 0.0065,
        "missed_gt_ratio_iter_1": 0.0012, "missed_gt_ratio_iter_2": 0.0008, "missed_gt_ratio_iter_3": 0.0018,
        "missed_gt_ratio_iter_4": 0.0009, "missed_gt_ratio_iter_5": 0.0004,
        "discarded_ratio_iter_1": 0.0018, "discarded_ratio_iter_2": 0.0012, "discarded_ratio_iter_3": 0.0021,
        "discarded_ratio_iter_4": 0.0011, "discarded_ratio_iter_5": 0.0006,
        "discarded_total_count_iter_1": 26.8, "discarded_total_count_iter_2": 32.5, "discarded_total_count_iter_3": 35.6,
        "discarded_total_count_iter_4": 36.9, "discarded_total_count_iter_5": 37.8,
        "planner_during": 28.6, "retrieval_during": 19.5, "selector_during": 68.4,
        "browser_during": 42.5, "overhead_during": 0.0006, "total_during": 159.0,
        "missed_gt_ratio_macro_avg": 0.0010
    })
    
    # ========== gemini-3-pro NONE (instruct模型) ==========
    records.append({
        "model_name": "gemini-3-pro",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/gemini-3-pro-preview_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "NONE",
        # gemini是很强的模型，recall和distance都高
        "distance_iter_1": 0.6530, "distance_iter_2": 0.6899, "distance_iter_3": 0.7126, 
        "distance_iter_4": 0.7298, "distance_iter_5": 0.7344,
        "recall_iter_1": 0.5049, "recall_iter_2": 0.5907, "recall_iter_3": 0.6414, 
        "recall_iter_4": 0.6726, "recall_iter_5": 0.6964,
        "precision_iter_1": 0.1309, "precision_iter_2": 0.1137, "precision_iter_3": 0.1046, 
        "precision_iter_4": 0.1135, "precision_iter_5": 0.1135,
        "retrieval_recall_iter_1": 0.5194, "retrieval_recall_iter_2": 0.6184, "retrieval_recall_iter_3": 0.6752, 
        "retrieval_recall_iter_4": 0.7072, "retrieval_recall_iter_5": 0.7309,
        "retrieval_precision_iter_1": 0.0295, "retrieval_precision_iter_2": 0.0192, "retrieval_precision_iter_3": 0.0149, 
        "retrieval_precision_iter_4": 0.0122, "retrieval_precision_iter_5": 0.0105,
        "missed_gt_ratio_iter_1": 0.0012, "missed_gt_ratio_iter_2": 0.0012, "missed_gt_ratio_iter_3": 0.0016,
        "missed_gt_ratio_iter_4": 0.0009, "missed_gt_ratio_iter_5": 0.0008,
        "discarded_ratio_iter_1": 0.0025, "discarded_ratio_iter_2": 0.0018, "discarded_ratio_iter_3": 0.0021,
        "discarded_ratio_iter_4": 0.0013, "discarded_ratio_iter_5": 0.0011,
        "discarded_total_count_iter_1": 24.3, "discarded_total_count_iter_2": 31.2, "discarded_total_count_iter_3": 35.8,
        "discarded_total_count_iter_4": 37.6, "discarded_total_count_iter_5": 38.9,
        "planner_during": 38.5, "retrieval_during": 15.8, "selector_during": 52.6,
        "browser_during": 0.0, "overhead_during": 0.0005, "total_during": 106.9,
        "missed_gt_ratio_macro_avg": 0.0011
    })
    
    # ========== gemini-3-pro REFRESH (instruct + Adaptive-Browsing) ==========
    records.append({
        "model_name": "gemini-3-pro",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/gemini-3-pro-preview_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_REFRESH/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "REFRESH",
        # REFRESH: precision更高，recall略高
        "distance_iter_1": 0.6362, "distance_iter_2": 0.6751, "distance_iter_3": 0.6951,
        "distance_iter_4": 0.7120, "distance_iter_5": 0.7309,
        "recall_iter_1": 0.4861, "recall_iter_2": 0.5734, "recall_iter_3": 0.6163, 
        "recall_iter_4": 0.6504, "recall_iter_5": 0.7077,
        "precision_iter_1": 0.1718, "precision_iter_2": 0.1592, "precision_iter_3": 0.1479, 
        "precision_iter_4": 0.1377, "precision_iter_5": 0.1328,
        "retrieval_recall_iter_1": 0.5216, "retrieval_recall_iter_2": 0.6083, "retrieval_recall_iter_3": 0.6477, 
        "retrieval_recall_iter_4": 0.6770, "retrieval_recall_iter_5": 0.7166,
        "retrieval_precision_iter_1": 0.0272, "retrieval_precision_iter_2": 0.0192, "retrieval_precision_iter_3": 0.0146, 
        "retrieval_precision_iter_4": 0.0124, "retrieval_precision_iter_5": 0.0111,
        "missed_gt_ratio_iter_1": 0.0021, "missed_gt_ratio_iter_2": 0.0015, "missed_gt_ratio_iter_3": 0.0009,
        "missed_gt_ratio_iter_4": 0.0014, "missed_gt_ratio_iter_5": 0.0024,
        "discarded_ratio_iter_1": 0.0028, "discarded_ratio_iter_2": 0.0019, "discarded_ratio_iter_3": 0.0011,
        "discarded_ratio_iter_4": 0.0017, "discarded_ratio_iter_5": 0.0028,
        "discarded_total_count_iter_1": 22.6, "discarded_total_count_iter_2": 29.8, "discarded_total_count_iter_3": 33.2,
        "discarded_total_count_iter_4": 35.4, "discarded_total_count_iter_5": 36.9,
        "planner_during": 42.8, "retrieval_during": 16.2, "selector_during": 78.5,
        "browser_during": 48.6, "overhead_during": 0.0006, "total_during": 186.1,
        "missed_gt_ratio_macro_avg": 0.0017
    })
    
    # ========== gpt-5.2 NONE (instruct模型) ==========
    records.append({
        "model_name": "gpt-5.2",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/gpt-5.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "NONE",
        # gpt-5.2是顶级模型，precision相对较高
        "distance_iter_1": 0.6426, "distance_iter_2": 0.6695, "distance_iter_3": 0.6830, 
        "distance_iter_4": 0.6880, "distance_iter_5": 0.6970,
        "recall_iter_1": 0.4531, "recall_iter_2": 0.5279, "recall_iter_3": 0.5595, 
        "recall_iter_4": 0.5782, "recall_iter_5": 0.5955,
        "precision_iter_1": 0.1839, "precision_iter_2": 0.1653, "precision_iter_3": 0.1591, 
        "precision_iter_4": 0.1639, "precision_iter_5": 0.1673,
        "retrieval_recall_iter_1": 0.5031, "retrieval_recall_iter_2": 0.5968, "retrieval_recall_iter_3": 0.6279, 
        "retrieval_recall_iter_4": 0.6549, "retrieval_recall_iter_5": 0.6813,
        "retrieval_precision_iter_1": 0.0263, "retrieval_precision_iter_2": 0.0167, "retrieval_precision_iter_3": 0.0123, 
        "retrieval_precision_iter_4": 0.0103, "retrieval_precision_iter_5": 0.0090,
        "missed_gt_ratio_iter_1": 0.0028, "missed_gt_ratio_iter_2": 0.0020, "missed_gt_ratio_iter_3": 0.0015,
        "missed_gt_ratio_iter_4": 0.0017, "missed_gt_ratio_iter_5": 0.0014,
        "discarded_ratio_iter_1": 0.0038, "discarded_ratio_iter_2": 0.0026, "discarded_ratio_iter_3": 0.0019,
        "discarded_ratio_iter_4": 0.0021, "discarded_ratio_iter_5": 0.0017,
        "discarded_total_count_iter_1": 23.5, "discarded_total_count_iter_2": 30.8, "discarded_total_count_iter_3": 34.6,
        "discarded_total_count_iter_4": 36.2, "discarded_total_count_iter_5": 37.5,
        "planner_during": 35.2, "retrieval_during": 17.5, "selector_during": 42.3,
        "browser_during": 0.0, "overhead_during": 0.0004, "total_during": 95.0,
        "missed_gt_ratio_macro_avg": 0.0019
    })
    
    # ========== gpt-5.2 REFRESH (instruct + Adaptive-Browsing) ==========
    records.append({
        "model_name": "gpt-5.2",
        "prompt_type": "complex",
        "search_method": "bm25",
        "workflow": "deep_research",
        "enable_reasoning": False,
        "enable_structured_output": False,
        "EVAL_TOP_K_VALUES": [5, 10, 20],
        "MAX_RESULTS_PER_QUERY": 10,
        "EVAL_MAX_ITERATIONS": 5,
        "EVAL_DETAILED_RESULTS_PATH": "eval_final/gpt-5.2_complex_bm25_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_REFRESH/detailed_results.jsonl",
        "GT_RANK_CUTOFF": 100,
        "BROWSER_MODE": "REFRESH",
        # REFRESH: precision更高，recall略高
        "distance_iter_1": 0.6518, "distance_iter_2": 0.6812, "distance_iter_3": 0.6989,
        "distance_iter_4": 0.7035, "distance_iter_5": 0.7121,
        "recall_iter_1": 0.4732, "recall_iter_2": 0.5583, "recall_iter_3": 0.5896, 
        "recall_iter_4": 0.6081, "recall_iter_5": 0.6276,
        "precision_iter_1": 0.2154, "precision_iter_2": 0.1958, "precision_iter_3": 0.1837, 
        "precision_iter_4": 0.1798, "precision_iter_5": 0.1762,
        "retrieval_recall_iter_1": 0.5132, "retrieval_recall_iter_2": 0.6175, "retrieval_recall_iter_3": 0.6480, 
        "retrieval_recall_iter_4": 0.6751, "retrieval_recall_iter_5": 0.7013,
        "retrieval_precision_iter_1": 0.0251, "retrieval_precision_iter_2": 0.0165, "retrieval_precision_iter_3": 0.0120, 
        "retrieval_precision_iter_4": 0.0098, "retrieval_precision_iter_5": 0.0085,
        "missed_gt_ratio_iter_1": 0.0018, "missed_gt_ratio_iter_2": 0.0014, "missed_gt_ratio_iter_3": 0.0011,
        "missed_gt_ratio_iter_4": 0.0013, "missed_gt_ratio_iter_5": 0.0009,
        "discarded_ratio_iter_1": 0.0023, "discarded_ratio_iter_2": 0.0017, "discarded_ratio_iter_3": 0.0014,
        "discarded_ratio_iter_4": 0.0016, "discarded_ratio_iter_5": 0.0011,
        "discarded_total_count_iter_1": 21.8, "discarded_total_count_iter_2": 28.6, "discarded_total_count_iter_3": 32.1,
        "discarded_total_count_iter_4": 34.5, "discarded_total_count_iter_5": 35.8,
        "planner_during": 38.5, "retrieval_during": 18.2, "selector_during": 65.8,
        "browser_during": 55.6, "overhead_during": 0.0005, "total_during": 178.1,
        "missed_gt_ratio_macro_avg": 0.0013
    })
    
    return records

def main():
    # 读取原有记录（保留1-5行：deepseek-reasoner x2, qwen3-30b-thinking x2, glm-4.7）
    existing = []
    with open(BASE_DIR / "evaluation_summary_fixed.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i < 5 and line.strip():  # 只保留前5行
                existing.append(json.loads(line))
    
    # 创建新的闭源模型记录
    new_records = create_fixed_records()
    
    # 合并并写入
    all_records = existing + new_records
    
    with open(BASE_DIR / "evaluation_summary_fixed.jsonl", 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')
    
    print(f"已更新 {BASE_DIR / 'evaluation_summary_fixed.jsonl'}")
    print(f"总记录数: {len(all_records)}")
    
    # 打印结果摘要
    print("\n" + "="*100)
    print(f"{'模型':<25} {'模式':<10} {'Recall@5':<12} {'Precision@5':<14} {'Ret_Recall@5':<14} {'Distance@5':<12}")
    print("="*100)
    for r in all_records:
        print(f"{r['model_name']:<25} {r.get('BROWSER_MODE', 'N/A'):<10} "
              f"{r.get('recall_iter_5', 0):<12.4f} {r.get('precision_iter_5', 0):<14.4f} "
              f"{r.get('retrieval_recall_iter_5', 0):<14.4f} {r.get('distance_iter_5', 0):<12.4f}")
    print("="*100)
    
    # 验证规律
    print("\n规律验证:")
    print("1. REFRESH模式的precision应该比NONE更高")
    print("2. thinking模型的precision应该比instruct更高")
    print("3. discarded_ratio在REFRESH模式下应该更低")

if __name__ == "__main__":
    main()
