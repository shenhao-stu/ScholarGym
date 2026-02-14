import json
import os
import re
import zipfile
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def get_month_from_arxiv_id(arxiv_id: str) -> Optional[int]:
    """Extracts month from an arXiv ID."""
    _, month = get_date_from_arxiv_id(arxiv_id)
    return month

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

import requests
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List

class ArxivSearcher:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, 
                 request_delay: float = 3,
                 timeout: int = 30,
                 max_request_attempts: int = 3,
                 backoff_factor: float = 1.5):
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_request_attempts = max_request_attempts
        self.backoff_factor = backoff_factor
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            )
        }
        logger.info(f"ArxivSearcher initialized: delay={request_delay}s, timeout={timeout}s")

    def _make_request(self, url: str) -> Optional[str]:
        for attempt in range(1, self.max_request_attempts + 1):
            try:
                backoff_sleep = self.request_delay * (self.backoff_factor ** (attempt - 1))
                time.sleep(backoff_sleep)
                resp = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code in {429, 500, 502, 503, 504}:
                    logger.warning(f"Transient HTTP {resp.status_code} on attempt {attempt}/{self.max_request_attempts} for URL: {url}")
                    if attempt < self.max_request_attempts:
                        continue
                    return None
                logger.warning(f"HTTP {resp.status_code} for URL: {url}")
                return None
            except requests.RequestException as e:
                logger.error(f"Request error on attempt {attempt}/{self.max_request_attempts} for {url}: {type(e).__name__} - {e}")
                if attempt < self.max_request_attempts:
                    continue
                return None
            except Exception as e:
                logger.error(f"Request failed for {url}: {type(e).__name__} - {e}")
                return None

    def _parse_arxiv_response(self, xml_content: str) -> Optional[List[Dict]]:
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            if not entries:
                return None

            papers = []
            for entry in entries:
                entry_id = entry.find('atom:id', ns)
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)

                authors = [
                    author.find('atom:name', ns).text
                    for author in entry.findall('atom:author', ns)
                    if author.find('atom:name', ns) is not None
                ]

                categories = [
                    cat.get('term')
                    for cat in entry.findall('atom:category', ns)
                    if cat.get('term')
                ]

                pdf_url = next(
                    (link.get('href') for link in entry.findall('atom:link', ns)
                     if link.get('title') == 'pdf'),
                    None
                )

                arxiv_id = entry_id.text.split('/')[-1] if entry_id is not None else None

                papers.append({
                    'id': arxiv_id,
                    'title': TextCleaner.clean_text(title.text if title is not None else ''),
                    'url': pdf_url,
                    'date': published.text[:10] if published is not None else '',
                    'abstract': TextCleaner.clean_text(summary.text if summary is not None else ''),
                    'category': categories,
                    'authors': authors,
                })
            return papers
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
            return None

    def search_by_ids(self, arxiv_ids: List[str]) -> Optional[List[Dict]]:
        try:
            clean_ids = [re.sub(r'v\d+$', '', arxiv_id.strip()) for arxiv_id in arxiv_ids]
            id_list_param = ",".join(clean_ids)
            url = f"{self.BASE_URL}?id_list={id_list_param}&start=0&max_results={len(clean_ids)}"

            response_text = self._make_request(url)
            if response_text:
                papers_data = self._parse_arxiv_response(response_text)
                if papers_data:
                    return papers_data

            logger.warning(f"[❌] Papers not found by IDs: {clean_ids}")
            return None
        except Exception as e:
            logger.error(f"[❌] Error searching arXiv by IDs {arxiv_ids}: {e}")
            return None

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
    namelist_set = set(paper_db.namelist())
    
    def _process_pasa_item(item):
        arxiv_id, title = item
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

        if title_key in namelist_set:
            try:
                content = paper_db.read(title_key)
                data = json.loads(content.decode("utf-8"))
                paper_data['abstract'] = data.get('abstract', '')
                paper_data['title'] = data.get('title', paper_data['title']).replace("\n", " ")
            except Exception as e:
                 logger.warning(f"Failed to process {title_key} for {arxiv_id}: {e}")
        
        return arxiv_id, paper_data

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_pasa_item, item) for item in id2paper.items()]
        for future in tqdm(as_completed(futures), total=len(id2paper), desc="Processing PASA papers"):
            arxiv_id, paper_data = future.result()
            if arxiv_id:
                papers[arxiv_id] = paper_data

    logger.info(f"Loaded {len(papers)} papers from PASA dataset")
    return papers


def load_litsearch_papers(litsearch_dir: str, corpus_id_to_arxiv_id_map: Dict[str, str]) -> Dict[str, Dict]:
    """Load papers from a local LitSearch dataset directory."""
    papers = {}
    corpus_path = os.path.join(litsearch_dir, "corpus_clean")

    logger.info(f"Loading LitSearch papers from huggingface: {corpus_path}")
    dataset = load_dataset("princeton-nlp/LitSearch", "corpus_clean", cache_dir=corpus_path)
    
    def _process_litsearch_row(row, corpus_id_to_arxiv_id_map):
        corpus_id = str(row.get('corpusid'))
        if not corpus_id:
            return None, None

        arxiv_id = corpus_id_to_arxiv_id_map.get(corpus_id)
        if arxiv_id:
            year, month = get_date_from_arxiv_id(arxiv_id)
            date = datetime(year, month, 1).strftime('%Y-%m-%d') if year and month else ''
            return arxiv_id, {
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
        else:
            # logger.warning(f"Could not find arxiv_id for corpusid {corpus_id} in corpus_clean.")
            return None, None

    with ThreadPoolExecutor() as executor:
        future_to_row = {executor.submit(_process_litsearch_row, row, corpus_id_to_arxiv_id_map): row for row in dataset['full']}
        
        for future in tqdm(as_completed(future_to_row), total=len(dataset['full']), desc="Processing LitSearch papers"):
            try:
                arxiv_id, paper_data = future.result()
                if arxiv_id and paper_data:
                    papers[arxiv_id] = paper_data
            except Exception as e:
                logger.error(f"Error processing row: {e}")


    logger.info(f"Loaded {len(papers)} papers from LitSearch dataset")
    return papers


def enrich_paper_data(
    papers: Dict[str, Dict], 
    output_dir: str, 
    batch_size: int = 10,
    request_delay: float = 1,
    num_retries: int = 0,
    save_every_n_batches: int = 10 
) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Enrich paper data synchronously using ArxivSearcher with checkpointing.
    """
    logger.info(f"Enriching data for {len(papers)} papers from arXiv with batch_size={batch_size}...")

    checkpoint_path = os.path.join(output_dir, "enrich_checkpoint_final.json")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load from checkpoint if exists ---
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

    logger.info(
        f"Loaded {len(enriched_from_checkpoint)} enriched papers and "
        f"{len(not_found_from_checkpoint)} not-found IDs from checkpoint. "
        f"{len(papers_to_process_ids)} papers remaining to process."
    )

    # --- Process remaining papers ---
    enriched_papers = enriched_from_checkpoint
    not_found_arxiv_ids = not_found_from_checkpoint

    searcher = ArxivSearcher(request_delay=request_delay)

    if papers_to_process_ids:
        paper_batches = [
            papers_to_process_ids[i:i + batch_size]
            for i in range(0, len(papers_to_process_ids), batch_size)
        ]

        for i, batch_ids in enumerate(tqdm(paper_batches, desc="Enriching paper data in batches")):
            logger.info(f"Processing batch {i+1}/{len(paper_batches)} with {len(batch_ids)} papers...")
            results = searcher.search_by_ids(batch_ids)

            if results:
                returned_ids = set()
                for result in results:
                    if result and result.get('id'):
                        arxiv_id = result['id']
                        cleared_id =re.sub(r'v\d+$', '', arxiv_id.strip()) 
                        returned_ids.add(cleared_id)
                        original_paper = papers.get(cleared_id, {'id': cleared_id})
                        merged_paper = original_paper.copy()
                        merged_paper.update(result)
                        enriched_papers[cleared_id] = merged_paper
                failed = set(batch_ids) - returned_ids
                not_found_arxiv_ids.update(failed)
            else:
                not_found_arxiv_ids.update(batch_ids)

            # --- Save checkpoint after each batch ---
            logger.info(f"Saving checkpoint after batch {i+1}")

            if (i + 1) % save_every_n_batches == 0 or i == len(paper_batches) - 1:
                logger.info(f"Saving checkpoint after batch {i+1}")
                tmp_path = checkpoint_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'enriched_papers': enriched_papers,
                        'not_found_arxiv_ids': list(not_found_arxiv_ids)
                    }, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, checkpoint_path)

    # --- Retry logic for not-found papers ---
    if not_found_arxiv_ids and num_retries > 0:
        logger.info(f"Initial enrichment done. Starting retry process for {len(not_found_arxiv_ids)} papers.")
        
        retry_request_delay = request_delay * 2
        retry_searcher = ArxivSearcher(request_delay=retry_request_delay)

        for attempt in range(num_retries):
            ids_to_retry = sorted(list(not_found_arxiv_ids))
            if not ids_to_retry:
                logger.info("All previously not-found papers have been found. Stopping retries.")
                break

            logger.info(f"--- Retry Attempt {attempt+1}/{num_retries} for {len(ids_to_retry)} papers ---")
            failed_in_retry = set()

            retry_batches = [
                ids_to_retry[j:j + batch_size]
                for j in range(0, len(ids_to_retry), batch_size)
            ]

            for i, batch_ids in enumerate(tqdm(retry_batches, desc="Retrying batches")):
                results = retry_searcher.search_by_ids(batch_ids)

                if results:
                    returned_ids = set()
                    for paper_data in results:
                        if paper_data and paper_data.get('id'):
                            arxiv_id = paper_data['id']
                            arxiv_id =re.sub(r'v\d+$', '', arxiv_id.strip()) 
                            returned_ids.add(arxiv_id)
                            original_paper = papers.get(arxiv_id, {'id': arxiv_id})
                            merged_paper = original_paper.copy()
                            merged_paper.update(paper_data)
                            enriched_papers[arxiv_id] = merged_paper
                    failed = set(batch_ids) - returned_ids
                    failed_in_retry.update(failed)
                else:
                    failed_in_retry.update(batch_ids)

            not_found_arxiv_ids = failed_in_retry

            # --- Save checkpoint after each retry attempt ---
            logger.info(f"Saving checkpoint after retry attempt {attempt+1}")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'enriched_papers': enriched_papers,
                    'not_found_arxiv_ids': list(not_found_arxiv_ids)
                }, f, ensure_ascii=False, indent=2)

        if not_found_arxiv_ids:
            logger.warning(f"After {num_retries} retries, {len(not_found_arxiv_ids)} papers still could not be found.")
        else:
            logger.info("All papers successfully enriched after retries.")

    # --- Final cleanup ---
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


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build unified benchmark dataset')
    parser.add_argument('--pasa_dir', type=str, default='pasa-dataset', help='Path to PASA dataset')
    parser.add_argument('--litsearch_dir', type=str, default='LitSearch', help='Path to LitSearch dataset')
    parser.add_argument('--output_dir', type=str, default='fast_benchmark_data', help='Output directory')
    parser.add_argument('--build_indices', action='store_true', default=True, help='Build BM25 and FAISS indices')
    parser.add_argument('--num_retries', type=int, default=0, help='Number of retries for papers that were not found initially')
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
                'date': p.get('date', ''),
                'abstract': p.get('abstract', ''),
                'arxiv_id': pid,
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
        rag_system.build_bm25_index(index_papers, 'fast_benchmark_data/bm25_index.pkl')
        rag_system.build_vector_library(index_papers, 'fast_benchmark_data/faiss_index')
        logger.info(f"Indices built and saved to fast_benchmark_data/bm25_index.pkl and fast_benchmark_data/faiss_index")

    logger.info("Benchmark dataset creation completed!")


if __name__ == "__main__":
    asyncio.run(main())
