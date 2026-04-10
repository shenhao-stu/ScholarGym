#!/usr/bin/env python3
"""
Citation RAG System for Paper Retrieval.
Supports BM25 keyword search and Qdrant vector search.
"""

import json
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import pickle
import os
import re
from logger import get_logger
import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

logger = get_logger(__name__, log_file='./log/rag.log')

class CitationRAGSystem:
    """
    RAG system for citation retrieval using Qdrant vector storage and BM25 keyword search.
    """

    def __init__(self, search_method: str = config.DEFAULT_SEARCH_METHOD):
        self.search_method = search_method
        self.bm25_index = None
        self.paper_metadata = {}
        # BM25 specific mappings
        self.bm25_corpus = None
        self.bm25_id_to_index = {}
        self.bm25_index_to_id = {}
        # Qdrant Vector Store
        self.qdrant_vector_store = None

    def load_or_build_indices(self, paper_db_path: str, bm25_path: str, rebuild: bool = False):
        """
        Load or build the Qdrant and BM25 indices as needed based on the search method.
        """
        needs_qdrant = self.search_method == 'vector'
        needs_bm25 = self.search_method == 'bm25'

        rebuild_bm25 = needs_bm25 and (rebuild or not os.path.exists(bm25_path))

        paper_db = None
        if rebuild_bm25:
            paper_db = self.load_paper_db(paper_db_path)

        if needs_qdrant:
            if rebuild:
                raise NotImplementedError(
                    "Qdrant index rebuild from eval is not supported. "
                    "Please build it manually:\n"
                    "  python code/build_vector_db.py --paper_db data/scholargym_paper_db.json"
                )
            else:
                logger.info("[✅]Loading existing Qdrant index...")
                self.load_qdrant_index()

        if needs_bm25:
            if rebuild_bm25:
                logger.info("[🔨]Building BM25 index...")
                if paper_db is None:
                    paper_db = self.load_paper_db(paper_db_path)
                self.build_bm25_index(paper_db, bm25_path)
            else:
                logger.info("[✅]Loading existing BM25 index...")
                self.load_bm25_index(bm25_path)

    def load_paper_db(self, paper_db_path: str) -> Dict[str, Dict]:
        """Load paper database from JSON file."""
        with open(paper_db_path, 'r', encoding='utf-8') as f:
            paper_db = json.load(f)

        logger.info(f"[✅]Loaded {len(paper_db)} paper database")
        return paper_db

    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing by tokenizing and normalizing."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def build_bm25_index(self, paper_db: Dict[str, Dict], save_path: str):
        """Build BM25 index from paper titles and abstracts."""
        logger.info("[🔨]Building BM25 index from titles and abstracts...")

        corpus = []
        paper_ids = []

        for paper_id, paper_data in tqdm(paper_db.items(), desc="Processing papers for BM25"):
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')

            if title or abstract:
                combined_text = f"title: {title} abstract: {abstract}"
                tokenized_doc = self._preprocess_text_for_bm25(combined_text)

                if tokenized_doc:
                    corpus.append(tokenized_doc)
                    paper_ids.append(paper_id)

                    if paper_id not in self.paper_metadata:
                        self.paper_metadata[paper_id] = {
                            'title': paper_data.get('title', ''),
                            'authors': paper_data.get('authors', []),
                            'date': paper_data.get('date', ''),
                            'url': paper_data.get('url', ''),
                            'abstract': paper_data.get('abstract', ''),
                            'arxiv_id': paper_data.get('arxiv_id', '')
                        }

        logger.info(f"Processing {len(corpus)} documents for BM25 indexing")

        self.bm25_index = BM25Okapi(corpus)
        self.bm25_corpus = corpus
        self.bm25_id_to_index = {paper_id: i for i, paper_id in enumerate(paper_ids)}
        self.bm25_index_to_id = {i: paper_id for i, paper_id in enumerate(paper_ids)}

        with open(save_path, 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'bm25_corpus': self.bm25_corpus,
                'paper_metadata': {pid: self.paper_metadata[pid] for pid in paper_ids if pid in self.paper_metadata},
                'id_to_index': self.bm25_id_to_index,
                'index_to_id': self.bm25_index_to_id
            }, f)

        logger.info(f"[✅]BM25 index and metadata saved to {save_path}")

    def load_bm25_index(self, index_path: str):
        """Load pre-built BM25 index and metadata."""
        logger.info(f"Loading BM25 index from {index_path}")

        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.bm25_corpus = data['bm25_corpus']
            self.paper_metadata.update(data['paper_metadata'])
            self.bm25_id_to_index = data['id_to_index']
            self.bm25_index_to_id = data['index_to_id']

        logger.info(f"Loaded BM25 index with {len(self.bm25_corpus)} documents")

    def load_qdrant_index(self):
        """Load pre-built Qdrant vector store."""
        logger.info(f"Loading Qdrant index from {config.QDRANT_URL}")

        embeddings = OllamaEmbeddings(
            model=config.QDRANT_EMBEDDING_MODEL,
            base_url=config.OLLAMA_URL
        )

        client = QdrantClient(url=config.QDRANT_URL)

        self.qdrant_vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )

    def search_citations_vector(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        before_date: Optional[str] = None,
        gt_arxiv_ids: Optional[Set[str]] = None,
        debug: bool = True,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ):
        if self.qdrant_vector_store is None:
             raise ValueError("Vector store not loaded.")

        multiplier = 5
        fetch_k = (offset + config.GT_RANK_CUTOFF) * multiplier

        # 1. Retrieval
        raw_results = self.qdrant_vector_store.similarity_search_with_score(
            query=query,
            k=fetch_k
        )

        # 2. Filtering
        filtered_results = []
        exclude_set = set(exclude_arxiv_ids) if exclude_arxiv_ids else set()

        for doc, score in raw_results:
            doc_meta = doc.metadata
            paper_id = doc_meta.get('arxiv_id') or doc_meta.get('id')
            doc_date = doc_meta.get('date')

            if paper_id and paper_id in exclude_set:
                continue

            if before_date and doc_date:
                if doc_date[:7] > before_date[:7]:
                    continue

            if not paper_id:
                continue

            filtered_results.append((paper_id, float(score), doc_meta))

        # 3. Ranking Metrics
        rank_dict = {}
        total_rank = max(len(filtered_results) - 1, 0)

        if debug and gt_arxiv_ids:
            found_arxiv_ids = set()
            for rank, (p_id, score, meta) in enumerate(filtered_results):
                if p_id in gt_arxiv_ids:
                    found_arxiv_ids.add(p_id)
                    rank_dict[p_id] = {
                        "rank": rank,
                        "total": total_rank
                    }

            for arxiv_id in gt_arxiv_ids:
                if arxiv_id not in found_arxiv_ids:
                    rank_dict[arxiv_id] = {
                        "rank": config.TOTAL_PAPER_NUM,
                        "total": total_rank
                    }

        # 4. Pagination
        paginated_results = filtered_results[offset : offset + top_k]

        return paginated_results, rank_dict

    def search_citations_bm25(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        before_date: Optional[str] = None,
        gt_arxiv_ids: Optional[Set[str]] = None,
        debug: bool = True,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Tuple[str, float, Dict]], Dict]:
        """Search for relevant citations using BM25, with optional time filtering."""
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded. Please load the index first.")

        tokenized_query = self._preprocess_text_for_bm25(query)
        if not tokenized_query:
            return [], {}

        scores = self.bm25_index.get_scores(tokenized_query)

        positive_indices = np.where(scores > 0)[0]

        # Apply time filter
        if before_date is not None:
            filtered_indices = []
            for idx in positive_indices:
                paper_id = self.bm25_index_to_id.get(idx)
                if paper_id:
                    paper_info = self.paper_metadata.get(paper_id, {})
                    paper_date = paper_info.get('date')
                    if paper_date and (paper_date[:7] <= before_date[:7]):
                        filtered_indices.append(idx)
            positive_indices = np.array(filtered_indices)

        sorted_indices = sorted(positive_indices, key=lambda i: scores[i], reverse=True)

        # Exclude specified arXiv IDs
        if exclude_arxiv_ids:
            exclude_set = set(exclude_arxiv_ids)
            filtered_sorted_indices = []
            for idx in sorted_indices:
                paper_id = self.bm25_index_to_id.get(idx)
                if not paper_id:
                    continue
                paper_info = self.paper_metadata.get(paper_id, {})
                arxiv_id = paper_info.get('arxiv_id', None)
                if arxiv_id in exclude_set:
                    continue
                filtered_sorted_indices.append(idx)
            sorted_indices = filtered_sorted_indices

        rank_dict = {}
        if debug and gt_arxiv_ids:
            found_arxiv_ids = set()
            total_rank = max(len(sorted_indices) - 1, 0)
            for rank, idx in enumerate(sorted_indices):
                paper_id = self.bm25_index_to_id.get(idx)
                if paper_id:
                    paper_info = self.paper_metadata.get(paper_id, {})
                    arxiv_id = paper_info.get('arxiv_id', None)
                    if arxiv_id in gt_arxiv_ids:
                        found_arxiv_ids.add(arxiv_id)
                        rank_dict[arxiv_id] = {
                            "rank": rank,
                            "total": total_rank
                        }

            for arxiv_id in gt_arxiv_ids:
                if arxiv_id not in found_arxiv_ids:
                    rank_dict[arxiv_id] = {
                        "rank": config.TOTAL_PAPER_NUM,
                        "total": total_rank
                    }

        paginated_indices = sorted_indices[offset : offset + top_k]

        results = []
        for idx in paginated_indices:
            paper_id = self.bm25_index_to_id[idx]
            paper_info = self.paper_metadata[paper_id]
            results.append((paper_id, float(scores[idx]), paper_info))

        return results, rank_dict

    def batch_search_citations_bm25(
        self,
        queries: List[str],
        top_k: int = 10,
        offset: int = 0,
        before_date: Optional[str] = None,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Perform batch BM25 search for multiple queries."""
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded.")

        return [
            self.search_citations_bm25(
                query, top_k, offset, before_date,
                exclude_arxiv_ids=exclude_arxiv_ids,
            )
            for query in queries
        ]

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """Get paper metadata by paper ID."""
        return self.paper_metadata.get(paper_id)

    def get_total_papers(self) -> int:
        """Get total number of papers in the index."""
        return len(self.paper_metadata)

    def is_loaded(self) -> bool:
        """Check if the vector store is loaded and ready."""
        return self.qdrant_vector_store is not None

    def is_bm25_loaded(self) -> bool:
        """Check if the BM25 system is loaded and ready."""
        return self.bm25_index is not None

    def get_available_search_methods(self) -> List[str]:
        """Get list of available search methods based on loaded indices."""
        methods = []
        if self.is_loaded():
            methods.append("vector")
        if self.is_bm25_loaded():
            methods.append("bm25")
        return methods
