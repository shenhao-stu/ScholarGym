#!/usr/bin/env python3
"""
Build a unified benchmark dataset from PASA and LitSearch datasets.
This script:
1. Loads queries from both datasets
2. Loads and deduplicates papers from both datasets, ensuring ground truth consistency
3. Creates a unified benchmark format compatible with eval.py
4. Builds BM25 and FAISS indices
"""
import json
import os
import re
import zipfile
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from logger import get_logger
from rag import CitationRAGSystem
import config
from datasets import load_dataset
import arxiv
from build import ArxivSearcher, TextCleaner

logger = get_logger(__name__, log_file='./log/build_benchmark.log')


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def get_date_from_arxiv_id(arxiv_id: str) -> Tuple[Optional[int], Optional[int]]:
    """Extracts year and month from an arXiv ID."""
    if not isinstance(arxiv_id, str):
        return None, None
    
    # New format: YYMM.NNNN... (e.g., 0704.1234)
    match = re.match(r'(\d{2})(\d{2})\.', arxiv_id)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if not (1 <= month <= 12):
            logger.warning(f"Invalid month {month} for {arxiv_id}")
            return None, None
        full_year = 2000 + year if year < 90 else 1900 + year
        return full_year, month
    
    # Old format: subject/YYMMNNN... (e.g., cs/0505003)
    match = re.match(r'[a-z\-]+/(?P<year>\d{2})(?P<month>\d{2})', arxiv_id)
    if match:
        year, month = int(match.group('year')), int(match.group('month'))
        if not (1 <= month <= 12):
            logger.warning(f"Invalid month {month} for {arxiv_id}")
            return None, None
        full_year = 2000 + year if year < 90 else 1900 + year
        return full_year, month
        
    return None, None

def get_year_from_arxiv_id(arxiv_id: str) -> Optional[int]:
    """Extracts year from an arXiv ID."""
    year, _ = get_date_from_arxiv_id(arxiv_id)
    return year

def get_latest_date_from_arxiv_ids(arxiv_ids: List[str]) -> str:
    """Extracts the latest year and month from a list of arXiv IDs."""
    latest_date = None
    for arxiv_id in arxiv_ids:
        year, month = get_date_from_arxiv_id(arxiv_id)
        if year and month:
            current_date = datetime(year, month, 1).strftime('%Y-%m')

            if latest_date is None or current_date > latest_date:
                latest_date = current_date
    
    return latest_date

def load_pasa_queries(pasa_dir: str) -> List[Dict]:
    """Load queries from PASA dataset (AutoScholarQuery and RealScholarQuery)."""
    queries = []
    
    # Load AutoScholarQuery
    auto_query_files = [
        # os.path.join(pasa_dir, "AutoScholarQuery", "train.jsonl"),
        os.path.join(pasa_dir, "AutoScholarQuery", "test.jsonl"),
        os.path.join(pasa_dir, "AutoScholarQuery", "dev.jsonl")
    ]
    
    for file_path in auto_query_files:
        if os.path.exists(file_path):
            logger.info(f"Loading queries from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    gt_ids = data.get('answer_arxiv_id', [])
                    query_obj = {
                        'query': data['question'],
                        'date': get_latest_date_from_arxiv_ids(gt_ids),
                        'answer_arxiv_ids': gt_ids,
                        'source': 'PASA_AutoScholar',
                        'qid': data['qid']
                    }
                    queries.append(query_obj)
    
    # Load RealScholarQuery
    real_query_file = os.path.join(pasa_dir, "RealScholarQuery", "test.jsonl")
    if os.path.exists(real_query_file):
        logger.info(f"Loading queries from {real_query_file}")
        with open(real_query_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                gt_ids = data.get('answer_arxiv_id', [])
                query_obj = {
                    'query': data['question'],
                    'date': get_latest_date_from_arxiv_ids(gt_ids),
                    'answer_arxiv_ids': gt_ids,
                    'source': 'PASA_RealScholar',
                    'qid': data['qid']
                }
                queries.append(query_obj)
    
    logger.info(f"Loaded {len(queries)} queries from PASA dataset")
    return queries


def build_litsearch_id_map(litsearch_dir: str) -> Dict[str, str]:
    """Build a mapping from corpusid to arxiv_id from LitSearch's s2orc data."""
    id_map = {}
    corpus_path = os.path.join(litsearch_dir, "corpus_s2orc")
    logger.info(f"Building corpusid to arxiv_id map from huggingface: {corpus_path}")
    
    try:
        dataset = load_dataset("princeton-nlp/LitSearch", "corpus_s2orc", cache_dir=corpus_path)
    except Exception as e:
        logger.error(f"Failed to load LitSearch s2orc dataset: {e}")
        return {}

    for row in tqdm(dataset['full'], desc="Building LitSearch ID map"):
        corpus_id = str(row.get('corpusid'))
        if not corpus_id:
            continue
            
        arxiv_id = None
        
        # Priority 1: Check externalids
        if row.get('externalids') and 'arxiv' in row['externalids'] and row['externalids']['arxiv']:
            arxiv_id = row['externalids']['arxiv']

        # Priority 2: Regex on pdfurls
        if not arxiv_id and row['content'].get('source') and row['content']['source'].get('pdfurls'):
            for url in row['content']['source']['pdfurls']:
                match = re.search(r'(\d{4}\.\d{4,5})', url)
                if match:
                    arxiv_id = match.group(1)
                    break
        
        if corpus_id and arxiv_id:
            id_map[corpus_id] = arxiv_id
        # else:
        #     logger.warning(f"Could not find arxiv_id for {corpus_id} in LitSearch")

    logger.info(f"Built mapping for {len(id_map)} LitSearch corpusids to arxiv_ids")
    return id_map


def load_litsearch_queries(litsearch_dir: str, corpus_id_to_arxiv_id_map: Dict[str, str]) -> List[Dict]:
    """Load queries from a local LitSearch dataset directory."""
    queries = []
    query_path = os.path.join(litsearch_dir, "query")

    logger.info(f"Loading LitSearch queries from huggingface: {query_path}")
    dataset = load_dataset("princeton-nlp/LitSearch", "query", cache_dir=query_path)
    for idx, row in enumerate(dataset['full']):
        flag = False
        corpus_ids = row.get('corpusids', [])
        if not isinstance(corpus_ids, list):
            corpus_ids = [corpus_ids]

        gt_ids = []
        for cid in corpus_ids:
            cid_str = str(cid)
            if cid_str in corpus_id_to_arxiv_id_map:
                gt_ids.append(corpus_id_to_arxiv_id_map[cid_str])
            else:
                flag = True
                # logger.warning(f"Could not find arxiv_id for corpusid {cid_str} in query LitSearch_{idx}")

        if flag:
            continue

        queries.append({
            'query': row.get('query', ''),
            'date': get_latest_date_from_arxiv_ids(gt_ids),
            'answer_arxiv_ids': gt_ids,
            'source': 'LitSearch',
            'qid': f"LitSearch_{idx}"
        })

    logger.info(f"Loaded {len(queries)} queries from LitSearch dataset")
    return queries


def load_pasa_papers(pasa_dir: str) -> Dict[str, Dict]:
    """Load papers from PASA dataset."""
    papers = {}
    id2paper_file = os.path.join(pasa_dir, "paper_database", "id2paper.json")
    paper_db_file = os.path.join(pasa_dir, "paper_database", "cs_paper_2nd.zip")

    logger.info(f"Loading PASA papers from {id2paper_file} and {paper_db_file}")
    with open(id2paper_file, 'r', encoding='utf-8') as f:
        id2paper = json.load(f)
    
    paper_db = zipfile.ZipFile(paper_db_file, "r")
    
    for arxiv_id, title in tqdm(id2paper.items(), desc="Processing PASA papers"):
        title_key = keep_letters(title)
        year, month = get_date_from_arxiv_id(arxiv_id)
        date = datetime(year, month, 1).strftime('%Y-%m-%d') if year and month else ''
        
        paper_data = {
            'id': arxiv_id,
            'arxiv_id': arxiv_id,
            'title': title.replace("\n", " "),
            'url': f"https://arxiv.org/abs/{arxiv_id}",
            'date': date,
            'abstract': '',
            'category': [],
            'authors': [],
            'source': 'PASA'
        }

        if title_key in paper_db.namelist():
            with paper_db.open(title_key) as f_in:
                data = json.loads(f_in.read().decode("utf-8"))
                paper_data['abstract'] = data.get('abstract', '')
                paper_data['title'] = data.get('title', paper_data['title']).replace("\n", " ")
        
        papers[arxiv_id] = paper_data

    logger.info(f"Loaded {len(papers)} papers from PASA dataset")
    return papers


def load_litsearch_papers(litsearch_dir: str, corpus_id_to_arxiv_id_map: Dict[str, str]) -> Dict[str, Dict]:
    """Load papers from a local LitSearch dataset directory."""
    papers = {}
    corpus_path = os.path.join(litsearch_dir, "corpus_clean")

    logger.info(f"Loading LitSearch papers from huggingface: {corpus_path}")
    dataset = load_dataset("princeton-nlp/LitSearch", "corpus_clean", cache_dir=corpus_path)
    
    for row in tqdm(dataset['full'], desc="Processing LitSearch papers"):
        corpus_id = str(row.get('corpusid'))
        if not corpus_id:
            continue

        arxiv_id = corpus_id_to_arxiv_id_map.get(corpus_id)
        if arxiv_id:
            year, month = get_date_from_arxiv_id(arxiv_id)
            date = datetime(year, month, 1).strftime('%Y-%m-%d') if year and month else ''
            
            papers[arxiv_id] = {
                'id': arxiv_id,
                'arxiv_id': arxiv_id,
                'title': row.get('title', ''),
                'url': f"https://arxiv.org/abs/{arxiv_id}",
                'date': date,
                'abstract': row.get('abstract', ''),
                'category': [],
                'authors': [],
                'source': 'LitSearch'
            }

    logger.info(f"Loaded {len(papers)} papers from LitSearch dataset")
    return papers


def enrich_paper_data(
    papers: Dict[str, Dict], 
    output_dir: str, 
    batch_size: int = 500,
    num_retries: int = 0
) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Enrich paper data synchronously using arxiv library with checkpointing and retries.
    """
    logger.info(f"Enriching data for {len(papers)} papers from arXiv with batch_size={batch_size}...")
    
    checkpoint_path = os.path.join(output_dir, "enrich_checkpoint.json")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            enriched_from_checkpoint = checkpoint_data.get('enriched_papers', {})
            not_found_from_checkpoint = set(checkpoint_data.get('not_found_arxiv_ids', []))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}. Starting fresh.")
            enriched_from_checkpoint, not_found_from_checkpoint = {}, set()
    else:
        enriched_from_checkpoint, not_found_from_checkpoint = {}, set()

    all_paper_ids = set(papers.keys())
    processed_ids = set(enriched_from_checkpoint.keys()) | not_found_from_checkpoint
    papers_to_process_ids = sorted(list(all_paper_ids - processed_ids))
    
    logger.info(f"Loaded {len(enriched_from_checkpoint)} enriched papers and {len(not_found_from_checkpoint)} not-found IDs from checkpoint. "
                f"{len(papers_to_process_ids)} papers remaining to process.")

    enriched_papers = enriched_from_checkpoint
    not_found_arxiv_ids = not_found_from_checkpoint

    # Initialize arxiv client/searcher once for both initial pass and retries
    arxiv_client = arxiv.Client(page_size=10, delay_seconds=0.01, num_retries=3)
    searcher = ArxivSearcher(arxiv_client)

    if papers_to_process_ids:
        paper_batches = [papers_to_process_ids[i:i + batch_size] for i in range(0, len(papers_to_process_ids), batch_size)]

        for i, batch_ids in enumerate(tqdm(paper_batches, desc="Enriching paper data in batches")):
            logger.info(f"Processing batch {i+1}/{len(paper_batches)} with {len(batch_ids)} papers...")
            
            for arxiv_id in tqdm(batch_ids, desc=f"Batch {i+1}", leave=False):
                original_paper = papers.get(arxiv_id, {'id': arxiv_id})
                try:
                    result = searcher.search_by_id(arxiv_id)
                    if result and result.get('id'):
                        merged_paper = original_paper.copy()
                        merged_paper.update(result)
                        enriched_papers[arxiv_id] = merged_paper
                    else:
                        not_found_arxiv_ids.add(arxiv_id)
                except Exception as e:
                    logger.error(f"Error processing {arxiv_id}: {e}")
                    not_found_arxiv_ids.add(arxiv_id)
            
            logger.info(f"Saving checkpoint after batch {i+1}")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'enriched_papers': enriched_papers,
                    'not_found_arxiv_ids': list(not_found_arxiv_ids)
                }, f, ensure_ascii=False, indent=2)

    if not_found_arxiv_ids and num_retries > 0:
        logger.info(f"Initial enrichment done. Starting retry process for {len(not_found_arxiv_ids)} papers.")
        
        for i in range(num_retries):
            ids_to_retry = sorted(list(not_found_arxiv_ids))
            if not ids_to_retry:
                logger.info("All previously not-found papers have been found. Stopping retries.")
                break

            logger.info(f"--- Retry Attempt {i+1}/{num_retries} for {len(ids_to_retry)} papers ---")
            
            failed_in_retry = set()

            retry_batches = [ids_to_retry[j:j + batch_size] for j in range(0, len(ids_to_retry), batch_size)]
            for b_idx, batch_ids in enumerate(tqdm(retry_batches, desc=f"Retrying attempt {i+1} in batches")):
                for arxiv_id in tqdm(batch_ids, desc=f"Retry batch {b_idx+1}", leave=False):
                    original_paper = papers.get(arxiv_id, {'id': arxiv_id})
                    try:
                        result = searcher.search_by_id(arxiv_id)
                        if result and result.get('id'):
                            merged_paper = original_paper.copy()
                            merged_paper.update(result)
                            enriched_papers[arxiv_id] = merged_paper
                        else:
                            failed_in_retry.add(arxiv_id)
                    except Exception as e:
                        logger.error(f"Retry error for {arxiv_id}: {e}")
                        failed_in_retry.add(arxiv_id)

                # Save checkpoint after each retry batch
                logger.info(f"Saving checkpoint after retry attempt {i+1}, batch {b_idx+1}/{len(retry_batches)}")
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'enriched_papers': enriched_papers,
                        'not_found_arxiv_ids': list(failed_in_retry)
                    }, f, ensure_ascii=False, indent=2)

            # Prepare for next retry attempt
            not_found_arxiv_ids = failed_in_retry

            # Also save a summary checkpoint at the end of the attempt
            logger.info(f"Saving checkpoint after retry attempt {i+1}")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'enriched_papers': enriched_papers,
                    'not_found_arxiv_ids': list(not_found_arxiv_ids)
                }, f, ensure_ascii=False, indent=2)

        if not_found_arxiv_ids:
            logger.warning(f"After {num_retries} retries, {len(not_found_arxiv_ids)} papers still could not be found.")
        else:
            logger.info("All papers successfully enriched after retries.")

    logger.info(f"Finished enriching data. Total enriched: {len(enriched_papers)}. Total not found: {len(not_found_arxiv_ids)}")
    
    final_papers = papers.copy()
    final_papers.update(enriched_papers)

    return final_papers, not_found_arxiv_ids


def deduplicate_and_merge_papers(pasa_papers: Dict, litsearch_papers: Dict) -> Tuple[Dict, Dict]:
    """Deduplicate papers based on arxiv_id and merge them."""
    logger.info("Deduplicating and merging papers based on arxiv_id...")
    
    merged_papers = {}
    
    # Process LitSearch papers first, as they are generally more complete
    for arxiv_id, paper in litsearch_papers.items():
        if arxiv_id:
            merged_papers[arxiv_id] = paper

    # Merge PASA papers, filling in missing information
    for arxiv_id, paper in pasa_papers.items():
        if arxiv_id:
            if arxiv_id not in merged_papers:
                merged_papers[arxiv_id] = paper
            else:
                # LitSearch data is preferred, but fill in missing fields from PASA
                for key, value in paper.items():
                    if key not in merged_papers[arxiv_id] or not merged_papers[arxiv_id][key]:
                        merged_papers[arxiv_id][key] = value

    logger.info(f"Merged and deduplicated papers down to {len(merged_papers)} unique arxiv_ids")
    return merged_papers

def convert_to_eval_format(queries: List[Dict], papers: Dict[str, Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Convert queries and papers to a format compatible with downstream evaluation."""
    eval_queries = []
    
    for query in queries:
        gt_papers = []
        for gt_id in query['answer_arxiv_ids']:
            if gt_id in papers:
                paper_info = papers[gt_id]
                year = get_year_from_arxiv_id(gt_id)
                gt_papers.append({
                    'arxiv_id': gt_id,
                    'title': paper_info.get('title', ''),
                    'year': year
                })

        eval_queries.append({
            'query': query['query'],
            'cited_paper': gt_papers,
            'gt_label': [1] * len(gt_papers),
            'date': query.get('date', ''),
            'source': query.get('source', ''),
            'qid': query.get('qid', ''),
            'valid': query.get('valid', False)
        })

    eval_papers = {
        paper_id: {
            'arxiv_id': data.get('arxiv_id', paper_id),
            'title': data.get('title', ''),
            'abstract': data.get('abstract', ''),
            'date': data.get('date', ''),
            'authors': data.get('authors', []),
            'category': data.get('category', [])
        }
        for paper_id, data in papers.items()
    }
    
    return eval_queries, eval_papers


def save_eval_data(queries: List[Dict], papers: Dict[str, Dict], output_dir: str):
    """Save benchmark data in the required format."""
    os.makedirs(output_dir, exist_ok=True)
    
    query_file = os.path.join(output_dir, "superlong_bench.jsonl")
    with open(query_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(queries)} queries to {query_file}")
    
    paper_file = os.path.join(output_dir, "superlong_paper_db.json")
    with open(paper_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(papers)} papers to {paper_file}")
    
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Total papers: {len(papers)}")


def load_or_process_data(args) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Loads pre-processed queries and papers from cache if available, otherwise processes them from source."""
    queries_cache_path = os.path.join(args.output_dir, "all_queries.jsonl")
    papers_cache_path = os.path.join(args.output_dir, "merged_papers.json")

    if os.path.exists(queries_cache_path) and os.path.exists(papers_cache_path):
        logger.info("Loading pre-processed queries and papers from cache...")
        all_queries = []
        with open(queries_cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_queries.append(json.loads(line))
        
        with open(papers_cache_path, 'r', encoding='utf-8') as f:
            merged_papers = json.load(f)
        logger.info(f"Loaded {len(all_queries)} queries and {len(merged_papers)} papers from cache.")
        return all_queries, merged_papers
    
    logger.info("No cache found. Processing queries and papers from source...")
    litsearch_id_map = build_litsearch_id_map(args.litsearch_dir)

    pasa_queries = load_pasa_queries(args.pasa_dir)
    litsearch_queries = load_litsearch_queries(args.litsearch_dir, litsearch_id_map)
    all_queries = pasa_queries + litsearch_queries

    pasa_papers = load_pasa_papers(args.pasa_dir)
    litsearch_papers = load_litsearch_papers(args.litsearch_dir, litsearch_id_map)
    
    merged_papers = deduplicate_and_merge_papers(pasa_papers, litsearch_papers)

    # Save to cache
    logger.info("Saving pre-processed queries and papers to cache...")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(queries_cache_path, 'w', encoding='utf-8') as f:
        for query in all_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    with open(papers_cache_path, 'w', encoding='utf-8') as f:
        json.dump(merged_papers, f, ensure_ascii=False, indent=2)
        
    return all_queries, merged_papers


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build unified benchmark dataset')
    parser.add_argument('--pasa_dir', type=str, default='pasa-dataset', help='Path to PASA dataset')
    parser.add_argument('--litsearch_dir', type=str, default='LitSearch', help='Path to LitSearch dataset')
    parser.add_argument('--output_dir', type=str, default='benchmark_data', help='Output directory')
    parser.add_argument('--build_indices', action='store_true', default=True, help='Build BM25 and FAISS indices')
    parser.add_argument('--num_retries', type=int, default=2, help='Number of retries for papers that were not found initially')
    args = parser.parse_args()

    all_queries, merged_papers = load_or_process_data(args)
    
    # Enrich merged papers with data from arXiv
    enriched_papers, not_found_arxiv_ids = enrich_paper_data(
        merged_papers, 
        args.output_dir,
        num_retries=args.num_retries
    )
    
    # Validate queries
    for query in all_queries:
        answer_ids = query.get('answer_arxiv_ids', [])
        if not answer_ids or any(aid in not_found_arxiv_ids for aid in answer_ids):
            query['valid'] = False
        else:
            query['valid'] = True
    
    eval_queries, eval_papers = convert_to_eval_format(all_queries, enriched_papers)
    save_eval_data(eval_queries, eval_papers, args.output_dir)
    
    if args.build_indices:
        logger.info("Building BM25 and FAISS indices...")
        
        # Adapt final papers to the format expected by RAG system
        index_papers = {
            pid: {
                'title': p.get('title', ''),
                'abstract': p.get('abstract', ''),
                'arxiv_id': pid,
                'date': p.get('date', ''),
                'authors': p.get('authors', []),
                'category': p.get('category', [])
            } 
            for pid, p in eval_papers.items()
        }

        rag_system = CitationRAGSystem(
            embedding_model_path=config.EMBEDDING_MODEL_PATH,
            device=config.DEVICE,
            search_method='both'
        )
        rag_system.build_bm25_index(index_papers, config.BM25_PATH)
        rag_system.build_vector_library(index_papers, config.FAISS_PATH_PREFIX)
        logger.info(f"Indices built and saved to {config.BM25_PATH} and {config.FAISS_PATH_PREFIX}")

    logger.info("Benchmark dataset creation completed!")

if __name__ == "__main__":
    main()