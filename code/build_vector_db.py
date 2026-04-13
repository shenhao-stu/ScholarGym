#!/usr/bin/env python3
"""
Build Qdrant vector database from paper corpus.

Usage:
    # Start Qdrant first:
    docker-compose up -d

    # Build with defaults from config.py:
    python code/build_vector_db.py --paper_db data/scholargym_paper_db.json

    # Override settings via CLI:
    python code/build_vector_db.py \
        --paper_db data/scholargym_paper_db.json \
        --qdrant_url http://localhost:6433 \
        --ollama_url http://localhost:11434 \
        --embedding_model qwen3-embedding:0.6b \
        --batch_size 128
"""
import os
import json
import argparse
import uuid
import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import config

# Deterministic UUID namespace for Qdrant point IDs.
# uuid5(NAMESPACE, arxiv_id) always produces the same UUID for the same paper,
# making upsert idempotent and preventing duplicate points on re-indexing.
_QDRANT_NS = uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890")


def _deterministic_id(key: str) -> str:
    """Generate a deterministic UUID string from a paper key (arxiv_id or paper_id)."""
    return str(uuid.uuid5(_QDRANT_NS, key))


def load_paper_db(path):
    """Load paper database from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Paper DB not found at {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def query_vector_db(query_text="Machine Learning", k=3,
                    qdrant_url=None, ollama_url=None,
                    embedding_model=None, collection_name=None):
    """Test interface to query the built vector database."""
    qdrant_url = qdrant_url or config.QDRANT_URL
    ollama_url = ollama_url or config.OLLAMA_URL
    embedding_model = embedding_model or config.QDRANT_EMBEDDING_MODEL
    collection_name = collection_name or config.QDRANT_COLLECTION_NAME

    print(f"\n[Test] Querying Vector DB with: '{query_text}'")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)

    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    results = vector_store.similarity_search(query_text, k=k)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Title: {doc.metadata.get('title')}")
        print(f"  ArXiv ID: {doc.metadata.get('arxiv_id')}")
        print(f"  Snippet: {doc.page_content[:150].replace(chr(10), ' ')}...")

    client.close()


def build_vector_db(paper_db_path=None, qdrant_url=None, ollama_url=None,
                    embedding_model=None, collection_name=None,
                    batch_size=128, test_mode=False, test_limit=100):
    """
    Build Qdrant vector database from paper corpus.

    Args:
        paper_db_path: Path to paper DB JSON file
        qdrant_url: Qdrant server URL
        ollama_url: Ollama server URL
        embedding_model: Ollama embedding model name
        collection_name: Qdrant collection name
        batch_size: Number of documents per indexing batch
        test_mode: If True, only index first test_limit papers
        test_limit: Number of papers in test mode
    """
    paper_db_path = paper_db_path or config.PAPER_DB_PATH
    qdrant_url = qdrant_url or config.QDRANT_URL
    ollama_url = ollama_url or config.OLLAMA_URL
    embedding_model = embedding_model or config.QDRANT_EMBEDDING_MODEL
    collection_name = collection_name or config.QDRANT_COLLECTION_NAME

    # 1. Load Data
    print(f"[1/5] Loading paper DB from {paper_db_path}...")
    try:
        paper_db = load_paper_db(paper_db_path)
    except FileNotFoundError:
        print(f"Error: Could not find paper DB at {paper_db_path}")
        return

    print(f"      Loaded {len(paper_db)} papers.")

    all_items = list(paper_db.items())
    if test_mode:
        print(f"      [Test Mode] Using first {test_limit} papers only.")
        all_items = all_items[:test_limit]

    # Checkpoint Setup
    qdrant_dir = os.path.dirname(paper_db_path)
    checkpoint_path = os.path.join(qdrant_dir, "vector_db_checkpoint.json")

    indexed_keys = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            try:
                indexed_keys = set(json.load(f))
                print(f"      Loaded checkpoint: {len(indexed_keys)} papers already indexed.")
            except json.JSONDecodeError:
                print("      Warning: Checkpoint file corrupted, starting fresh.")

    # 2. Prepare Documents
    doc_list = []
    print("[2/5] Converting papers to Documents...")
    for paper_id, data in tqdm.tqdm(all_items, desc="Processing Papers"):
        arxiv_id = data.get('arxiv_id', '')
        checkpoint_key = arxiv_id if arxiv_id else paper_id

        if checkpoint_key in indexed_keys:
            continue

        title = data.get('title', '')
        abstract = data.get('abstract', '')

        if not title and not abstract:
            continue

        content = f"title: {title}\n abstract: {abstract}"
        raw_date = data.get('date', '')
        url = data.get('url', '')
        authors = data.get('authors', [])
        metadata = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "arxiv_id": arxiv_id,
            "date": raw_date,
            "url": url,
            "_checkpoint_key": arxiv_id if arxiv_id else paper_id
        }
        doc = Document(page_content=content, metadata=metadata)
        doc.id = _deterministic_id(checkpoint_key)
        doc_list.append(doc)

    print(f"      Prepared {len(doc_list)} new documents to index.")

    if not doc_list:
        print("No new documents to index. All done!")
        return

    # 3. Initialize Embedding & Qdrant
    print("[3/5] Initializing Vector DB (Server Mode)...")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)

    print(f"      Connecting to Qdrant Server at: {qdrant_url}")
    client = QdrantClient(url=qdrant_url)

    print("      Probing embedding dimension...")
    try:
        dummy_vec = embeddings.embed_query("test")
        vector_size = len(dummy_vec)
        print(f"      Vector Dimension: {vector_size}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            )
        )
        print(f"      Created collection: {collection_name}")
    else:
        print(f"      Collection {collection_name} already exists.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # 4. Indexing Loop
    print(f"[4/5] Indexing {len(doc_list)} vectors...")

    # Checkpoint write cadence: writing the full indexed_keys set every batch
    # is O(N^2) total I/O. With deterministic UUIDs, upsert is idempotent, so
    # losing a handful of batches on crash is safe — flush every N batches.
    CHECKPOINT_FLUSH_EVERY = 50
    batches_since_flush = 0

    def _flush_checkpoint():
        tmp_path = checkpoint_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(list(indexed_keys), f)
        os.replace(tmp_path, checkpoint_path)

    for i in tqdm.tqdm(range(0, len(doc_list), batch_size), desc="Indexing Batches"):
        batch = doc_list[i:i+batch_size]
        try:
            vector_store.add_documents(documents=batch)

            for doc in batch:
                indexed_keys.add(doc.metadata['_checkpoint_key'])

            batches_since_flush += 1
            if batches_since_flush >= CHECKPOINT_FLUSH_EVERY:
                _flush_checkpoint()
                batches_since_flush = 0

        except Exception as e:
            print(f"Error indexing batch {i}: {e}")

    # Final flush to persist the tail batches
    _flush_checkpoint()

    print("[5/5] Done! Vector DB built successfully.")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build Qdrant vector database from paper corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default config:
  python code/build_vector_db.py --paper_db data/scholargym_paper_db.json

  # Quick test with 100 papers:
  python code/build_vector_db.py --paper_db data/scholargym_paper_db.json --test_mode

  # Custom endpoints:
  python code/build_vector_db.py --paper_db data/scholargym_paper_db.json \\
      --qdrant_url http://localhost:6433 --ollama_url http://localhost:11434
"""
    )
    parser.add_argument('--paper_db', type=str, default=None, help=f'Path to paper DB JSON (default: {config.PAPER_DB_PATH})')
    parser.add_argument('--qdrant_url', type=str, default=None, help=f'Qdrant server URL (default: {config.QDRANT_URL})')
    parser.add_argument('--ollama_url', type=str, default=None, help=f'Ollama server URL (default: {config.OLLAMA_URL})')
    parser.add_argument('--embedding_model', type=str, default=None, help=f'Ollama embedding model (default: {config.QDRANT_EMBEDDING_MODEL})')
    parser.add_argument('--collection_name', type=str, default=None, help=f'Qdrant collection name (default: {config.QDRANT_COLLECTION_NAME})')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for indexing (default: 128)')
    parser.add_argument('--test_mode', action='store_true', help='Index only first 100 papers for testing')
    parser.add_argument('--test_query', type=str, default=None, help='Run a test query after building (e.g. "large language models")')

    args = parser.parse_args()

    build_vector_db(
        paper_db_path=args.paper_db,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
    )

    if args.test_query:
        query_vector_db(
            query_text=args.test_query,
            qdrant_url=args.qdrant_url,
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
            collection_name=args.collection_name,
        )
