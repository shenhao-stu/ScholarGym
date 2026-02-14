#!/usr/bin/env python3
"""
Citation RAG System for Vector-based Paper Retrieval
This module provides functionality to build FAISS vector libraries from paper abstracts
and perform similarity-based citation retrieval.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import pickle
import os
import re
from logger import get_logger
import glob
import config
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

logger = get_logger(__name__, log_file='./log/rag.log')

class CitationRAGSystem:
    """
    RAG system for citation retrieval using FAISS vector storage and BM25 keyword search.
    Handles vector library construction and similarity-based paper retrieval.
    """
    
    def __init__(self, embedding_model_path: str = config.EMBEDDING_MODEL_PATH, device: str = config.DEVICE, search_method: str = config.DEFAULT_SEARCH_METHOD):
        """
        Initialize the citation RAG system with embedding model.
        
        Args:
            embedding_model_path: Path to the embedding model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.embedding_model = SentenceTransformer(
            embedding_model_path, 
            trust_remote_code=True,
            device=device
        ) if search_method != 'bm25' else None
        self.search_method = search_method
        self.faiss_index = None
        self.bm25_index = None
        self.paper_metadata = {}
        # FAISS specific mappings
        self.faiss_id_to_index = {}
        self.faiss_index_to_id = {}
        # BM25 specific mappings
        self.bm25_corpus = None
        self.bm25_id_to_index = {}
        self.bm25_index_to_id = {}
        # Qdrant Vector Store
        self.qdrant_vector_store = None
        
    def load_or_build_indices(self, paper_db_path: str, faiss_path: str, bm25_path: str,rebuild: bool = False):
        """
        Load or build the FAISS and BM25 indices as needed based on the search method.
        """
        needs_qdrant = self.search_method in ['vector', 'hybrid']
        needs_bm25 = self.search_method in ['bm25', 'hybrid']

        rebuild_qdrant = needs_qdrant and rebuild
        rebuild_bm25 = needs_bm25 and (rebuild or not os.path.exists(bm25_path))

        paper_db = None
        if rebuild_qdrant or rebuild_bm25:
            paper_db = self.load_paper_db(paper_db_path)

        # qdrant 暂未集成重建，需提前手动构建
        if needs_qdrant:
            if rebuild_qdrant:
                # logger.info("[🔨]Building Qdrant index...")
                # self.build_vector_library(paper_db, faiss_path)
                logger.info("[🔨]Please build Qdrant index manually...")
                raise NotImplementedError("Qdrant index building not implemented. Please build it manually.")
            else:
                logger.info("[✅]Loading existing Qdrant index...")
                self.load_qdrant_index()
                
        
        if needs_bm25:
            if rebuild_bm25:
                logger.info("[🔨]Building BM25 index...")
                self.build_bm25_index(paper_db, bm25_path)
            else:
                logger.info("[✅]Loading existing BM25 index...")
                self.load_bm25_index(bm25_path)

    def load_paper_db(self, paper_db_path: str) -> Dict[str, Dict]:
        """
        Load paper database from JSON file.
        
        Args:
            paper_db_path: Path to the paper database JSON file
            
        Returns:
            Dictionary of paper ID to paper data
        """
        with open(paper_db_path, 'r', encoding='utf-8') as f:
            paper_db = json.load(f)
        
        logger.info(f"[✅]Loaded {len(paper_db)} paper database")
        return paper_db
    
    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing by tokenizing and normalizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        # Remove punctuation and split into tokens
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def build_bm25_index(self, paper_db: Dict[str, Dict], save_path: str):
        """
        Build BM25 index from paper titles and abstracts.
        
        Args:
            paper_db: Dictionary of paper data
            save_path: Path prefix to save the BM25 index and corpus
        """
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
        """
        Load pre-built BM25 index and metadata.
        
        Args:
            index_path: Path prefix of the saved BM25 index file
        """
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed. Please install with: pip install rank-bm25")
        
        logger.info(f"Loading BM25 index from {index_path}")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.bm25_corpus = data['bm25_corpus']
            self.paper_metadata.update(data['paper_metadata'])
            self.bm25_id_to_index = data['id_to_index']
            self.bm25_index_to_id = data['index_to_id']
        
        logger.info(f"Loaded BM25 index with {len(self.bm25_corpus)} documents")

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
        if not hasattr(self, 'qdrant_vector_store') or self.qdrant_vector_store is None:
             raise ValueError("Vector store not loaded.")

        # vector 的 top_k 检索有优化，搜索全部文章会很慢
        multiplier = 5
        fetch_k = (offset + config.GT_RANK_CUTOFF) * multiplier
        # fetch_k = config.TOTAL_PAPER_NUM
        
        
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
        """
        Search for relevant citations using BM25, with optional time filtering.
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded. Please load the index first.")
        
        tokenized_query = self._preprocess_text_for_bm25(query)
        if not tokenized_query:
            return [], {}
        
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get all results with a positive score
        positive_indices = np.where(scores > 0)[0]
        
        # Apply time filter if provided
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
            
        # Sort by score and apply pagination
        sorted_indices = sorted(positive_indices, key=lambda i: scores[i], reverse=True)

        # Exclude specified arXiv IDs (keep top_k by skipping during pagination)
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
            total_rank = max(len(sorted_indices) - 1, 0)  # total_rank is the max rank index
            for rank, idx in enumerate(sorted_indices):
                paper_id = self.bm25_index_to_id.get(idx)
                if paper_id:
                    paper_info = self.paper_metadata.get(paper_id, {})
                    arxiv_id = paper_info.get('arxiv_id', None)
                    if arxiv_id in gt_arxiv_ids:
                        found_arxiv_ids.add(arxiv_id)
                        # rank starts from 0
                        rank_dict[arxiv_id] = {
                            "rank": rank,
                            "total": total_rank
                        }

            # For gt_arxiv_ids that are not found, set rank to total paper count, score is 0
            for arxiv_id in gt_arxiv_ids:
                if arxiv_id not in found_arxiv_ids:
                    rank_dict[arxiv_id] = {
                        "rank": config.TOTAL_PAPER_NUM,  # Use total paper count for not found
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
        """
        Perform batch BM25 search for multiple queries efficiently.
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded. Please load or build the BM25 index first.")
        
        # Note: The date filter is applied individually per query in the loop.
        return [
            self.search_citations_bm25(
                query,
                top_k,
                offset,
                before_date,
                exclude_arxiv_ids=exclude_arxiv_ids,
            )
            for query in queries
        ]
    
    def batch_search(self, queries: List[str], top_k: int, offset: int = 0) -> List[List[Tuple]]:
        """
        Unified batch search that delegates to the appropriate method based on `self.search_method`.
        """
        if self.search_method == 'vector':
            return self.batch_search_citations(queries, top_k=top_k, offset=offset)
        elif self.search_method == 'bm25':
            return self.batch_search_citations_bm25(queries, top_k=top_k, offset=offset)
        elif self.search_method == 'hybrid':
            # Hybrid search supports offset via slicing after rerank.
            return [self.search_citations_hybrid(q, top_k=top_k, offset=offset) for q in queries]
        else:
            logger.error(f"Unknown search method: {self.search_method}")
            return []

    def search_citations_hybrid(self, query: str, top_k: int = config.HYBRID_SEARCH_TOP_K, 
                               vector_weight: float = config.HYBRID_VECTOR_WEIGHT, bm25_weight: float = config.HYBRID_BM25_WEIGHT, before_date: Optional[str] = None, offset: int = 0) -> List[Tuple[str, float, Dict]]:
        """
        Hybrid search combining FAISS vector similarity and BM25 keyword matching.
        Fetches a larger pool of candidates from each search method and then reranks.
        Supports pagination via `offset`.
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("Both FAISS and BM25 indices must be loaded for hybrid search.")
        
        # Fetch more candidates to allow offset and date filters
        candidate_pool_size = max(0, top_k + offset)
        candidate_pool_size = candidate_pool_size * 2 if candidate_pool_size > 0 else top_k * 2
        
        # Fetch candidates from both search methods
        vector_results = self.search_citations(query, candidate_pool_size, offset=0, before_date=before_date)
        bm25_results = self.search_citations_bm25(query, candidate_pool_size, offset=0, before_date=before_date)
        
        combined_scores = {}
        paper_info_map = {}
        
        # Normalize and combine vector scores
        if vector_results:
            vector_scores = [score for _, score, _ in vector_results]
            max_vector_score = max(vector_scores) if vector_scores else 1.0
            
            for paper_id, score, paper_info in vector_results:
                normalized_score = score / max_vector_score if max_vector_score > 0 else 0
                combined_scores[paper_id] = vector_weight * normalized_score
                paper_info_map[paper_id] = paper_info
        
        # Normalize and combine BM25 scores
        if bm25_results:
            bm25_scores = [score for _, score, _ in bm25_results]
            max_bm25_score = max(bm25_scores) if bm25_scores else 1.0
            
            for paper_id, score, paper_info in bm25_results:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                
                if paper_id in combined_scores:
                    combined_scores[paper_id] += bm25_weight * normalized_score
                else:
                    combined_scores[paper_id] = bm25_weight * normalized_score
                    paper_info_map[paper_id] = paper_info
        
        # Sort by combined score and apply offset and top_k
        sorted_papers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        sliced = sorted_papers[offset : offset + top_k]
        
        results = []
        for paper_id, score in sliced:
            results.append((paper_id, score, paper_info_map[paper_id]))
        
        return results

    def build_vector_library(self, paper_db: Dict[str, Dict], save_path: str, batch_size: int = config.EMBEDDING_BATCH_SIZE):
        """
        Build FAISS vector library from paper titles and abstracts.
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Initialize with a vector-based search method.")

        logger.info("[🔨]Building FAISS index from titles and abstracts...")
        
        texts_to_embed = []
        paper_ids = []
        
        for paper_id, paper_data in tqdm(paper_db.items(), desc="Processing papers for embedding"):
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')

            if title or abstract:
                combined_text = f"title: {title} abstract: {abstract}"
                texts_to_embed.append(combined_text)
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
        
        logger.info(f"Processing {len(texts_to_embed)} documents for embedding")
        
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(dimension)

        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating embeddings and building index"):
            batch_texts = texts_to_embed[i:i+batch_size]
            
            # Generate embeddings for the current batch
            batch_embeddings = self.embedding_model.encode(
                batch_texts, show_progress_bar=False, convert_to_numpy=True
            ).astype('float32')
            
            # Normalize and add to FAISS index
            faiss.normalize_L2(batch_embeddings)
            self.faiss_index.add(batch_embeddings)
        
        logger.info(f"Generated and indexed embeddings for {self.faiss_index.ntotal} documents.")
        
        self.faiss_id_to_index = {paper_id: i for i, paper_id in enumerate(paper_ids)}
        self.faiss_index_to_id = {i: paper_id for i, paper_id in enumerate(paper_ids)}
        
        faiss.write_index(self.faiss_index, f"{save_path}.bin")
        
        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'paper_metadata': {pid: self.paper_metadata[pid] for pid in paper_ids if pid in self.paper_metadata},
                'id_to_index': self.faiss_id_to_index,
                'index_to_id': self.faiss_index_to_id
            }, f)
        
        logger.info(f"FAISS index and metadata saved to {save_path}")
    
    def load_qdrant_index(self):
        """
        Load pre-built Qdrant index and metadata.
        """
        logger.info(f"Loading Qdrant index")
        
        embedding_model_name = "qwen3-embedding:0.6b"  
        print(f"Loading Embeddings: {embedding_model_name}...")
        embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=config.OLLAMA_URL)
        
        COLLECTION_NAME = "paper_knowledge_base"
        
        client = QdrantClient(url=config.QDRANT_URL)
        
        self.qdrant_vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    
    def load_vector_library(self, index_path: str):
        """
        Load pre-built FAISS index and metadata.
        """
        logger.info(f"Loading FAISS index from {index_path}")
        
        self.faiss_index = faiss.read_index(f"{index_path}.bin")
        
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.paper_metadata.update(data['paper_metadata'])
            self.faiss_id_to_index = data['id_to_index']
            self.faiss_index_to_id = data['index_to_id']
        
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
    
    def search_citations(
        self,
        query: str,
        top_k: int = config.VECTOR_SEARCH_TOP_K,
        offset: int = 0,
        before_date: Optional[str] = None,
        gt_arxiv_ids: Optional[Set[str]] = None,
        debug: bool = True
    ) -> Tuple[List[Tuple[str, float, Dict]], Dict]:
        """
        Search for relevant citations using vector similarity, with optional time filtering.
        Returns results and (optionally) a rank_dict if debug is enabled.
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded. Please load or build the index first.")
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Initialize with a vector-based search method.")
        
        # 编码查询向量
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # 获取候选数量
        candidate_k = (top_k + offset) * 5 if before_date is not None else (top_k + offset)
        if candidate_k > self.faiss_index.ntotal:
            candidate_k = self.faiss_index.ntotal
            
        similarities, indices = self.faiss_index.search(query_embedding, candidate_k)
        
        # 收集原始结果
        unfiltered_results = []
        for similarity, index in zip(similarities[0], indices[0]):
            if index != -1:
                paper_id = self.faiss_index_to_id[index]
                paper_info = self.paper_metadata[paper_id]
                unfiltered_results.append((paper_id, float(similarity), paper_info))

        # 时间过滤
        if before_date is not None:
            filtered_list = []
            for paper_id, similarity, paper_info in unfiltered_results:
                paper_date = paper_info.get('date')
                if paper_date and (paper_date[:7] <= before_date[:7]):
                    filtered_list.append((paper_id, similarity, paper_info))
            unfiltered_results = filtered_list

        # 排序（FAISS 已经返回按相似度降序，但过滤后要保证顺序一致）
        sorted_results = sorted(unfiltered_results, key=lambda x: x[1], reverse=True)

        # debug: rank_dict 部分
        rank_dict = {}
        if debug and gt_arxiv_ids:
            found_arxiv_ids = set()
            total_len = len(sorted_results) + 1
            for rank, (paper_id, similarity, paper_info) in enumerate(sorted_results):
                arxiv_id = paper_info.get('arxiv_id')
                if arxiv_id in gt_arxiv_ids:
                    found_arxiv_ids.add(arxiv_id)
                    actual_rank = rank + 1
                    if actual_rank > offset:
                        rank_value = actual_rank - offset
                    else:
                        rank_value = 0
                    rank_dict[arxiv_id] = {
                        "rank": rank_value,
                        "total": total_len
                    }
            # 没出现的 gt_arxiv_ids 记为 total_len
            for arxiv_id in gt_arxiv_ids:
                if arxiv_id not in found_arxiv_ids:
                    rank_dict[arxiv_id] = {
                        "rank": total_len,
                        "total": total_len
                    }

        # 应用分页
        paginated_results = sorted_results[offset : offset + top_k]
        
        return paginated_results, rank_dict


    def batch_search_citations(self, queries: List[str], top_k: int = 10, offset: int = 0, before_date: Optional[str] = None) -> List[List[Tuple[str, float, Dict]]]:
        """
        Perform batch search for multiple queries efficiently.
        """
        # Note: The date filter is applied individually per query in the loop.
        return [self.search_citations(q, top_k, offset=offset, before_date=before_date) for q in queries]
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Get paper metadata by paper ID.
        """
        return self.paper_metadata.get(paper_id)
    
    def get_total_papers(self) -> int:
        """
        Get total number of papers in the index.
        """
        return len(self.paper_metadata)
    
    def is_loaded(self) -> bool:
        """
        Check if the RAG system is loaded and ready.
        """
        return self.qdrant_vector_store is not None
    
    def is_bm25_loaded(self) -> bool:
        """
        Check if the BM25 system is loaded and ready.
        """
        return self.bm25_index is not None
    
    def get_available_search_methods(self) -> List[str]:
        """
        Get list of available search methods based on loaded indices.
        """
        methods = []
        if self.is_loaded():
            methods.append("vector")
        if self.is_bm25_loaded():
            methods.append("bm25")
        if self.is_loaded() and self.is_bm25_loaded():
            methods.append("hybrid")
        return methods

def merge_citation_files(input_dir: str, output_file: str, file_pattern: str = "ref_*.json"):
    """
    Merges multiple JSON citation files from a directory into a single JSON file.
    """
    logger.info(f"Merging citation files from '{input_dir}' into '{output_file}'")
    
    merged_citations = {}
    json_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not json_files:
        logger.warning(f"No files matching pattern '{file_pattern}' found in '{input_dir}'")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        return
        
    for file_path in tqdm(json_files, desc="Merging citation files"):
        file_basename = os.path.basename(file_path).replace('.json', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key, value in data.items():
                new_key = f"{file_basename}-{key}"
                if new_key in merged_citations:
                        logger.warning(f"[💢Failed]Generated duplicate key '{new_key}'. Overwriting.")
                merged_citations[new_key] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_citations, f, ensure_ascii=False)
        
    logger.info(f"Successfully merged {len(merged_citations)} citation entries into '{output_file}'")

def display_search_results(method_name: str, results: List[Tuple[str, float, Dict]]):
    """Helper function to print search results in a formatted way."""
    logger.info(f"####{method_name.upper()} Search Results####")
    if not results:
        logger.info("No results found.")
    for i, (paper_id, score, paper_info) in enumerate(results):
        logger.info(f"{i+1}. [Score: {score:.4f}] {paper_info.get('title', 'N/A')}")

def main():
    """
    Main function for building or loading vector and BM25 libraries.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Citation RAG System with Vector and BM25 Search')
    parser.add_argument('--cited_data_dir', type=str, default=config.CITED_DATA_DIR)
    parser.add_argument('--paper_db', type=str, default=config.PAPER_DB_PATH)
    parser.add_argument('--force_merge_citations', action='store_true')
    parser.add_argument('--embedding_model', type=str, default=config.EMBEDDING_MODEL_PATH)
    parser.add_argument('--faiss_path', type=str, default=config.FAISS_PATH_PREFIX, help='Path prefix to save/load FAISS index')
    parser.add_argument('--bm25_path', type=str, default=config.BM25_PATH, help='Path to save/load BM25 index')
    parser.add_argument('--qdrant_path', type=str, default=config.QDRANT_PATH, help='Path to save/load Qdrant index')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of all indices')
    parser.add_argument('--rebuild_faiss', action='store_true', help='Force rebuild of FAISS index only')
    parser.add_argument('--rebuild_bm25', action='store_true', help='Force rebuild of BM25 index only')
    parser.add_argument('--search_method', type=str, default=config.DEFAULT_SEARCH_METHOD, choices=['vector', 'bm25', 'both', 'hybrid'])
    parser.add_argument('--device', type=str, default=config.DEVICE, help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.force_merge_citations or not os.path.exists(args.paper_db):
        logger.info("[🔨]Consolidating citation JSON files...")
        merge_citation_files(args.cited_data_dir, args.paper_db)
    else:
        logger.info(f"[✅]Using existing consolidated citation file: {args.paper_db}")

    rag_system = CitationRAGSystem(embedding_model_path=args.embedding_model, device=args.device, search_method=args.search_method)
    
    rag_system.load_or_build_indices(
        paper_db_path=args.paper_db,
        faiss_path=args.faiss_path,
        bm25_path=args.bm25_path,
        rebuild=(args.rebuild or args.rebuild_faiss or args.rebuild_bm25)
    )
    
    available_methods = rag_system.get_available_search_methods()
    logger.info(f"[✅]RAG system ready with {rag_system.get_total_papers()} papers indexed")
    logger.info(f"[✅]Available search methods: {', '.join(available_methods)}")
    
    if available_methods:
        test_query = "machine learning evaluation metrics"
        logger.info(f"\n[🚀]Running test searches with query: '{test_query}'")
        
        if 'vector' in available_methods:
            vector_results = rag_system.search_citations_vector(test_query, top_k=3)
            display_search_results("vector", vector_results)
        
        if 'bm25' in available_methods:
            bm25_results = rag_system.search_citations_bm25(test_query, top_k=3)
            display_search_results("bm25", bm25_results)
        
        if 'hybrid' in available_methods:
            hybrid_results = rag_system.search_citations_hybrid(test_query, top_k=3)
            display_search_results("hybrid", hybrid_results)

if __name__ == "__main__":
    main()
