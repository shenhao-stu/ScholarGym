import os
import json
import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import config

# 🚀 修改 1: 定义服务地址配置
QDRANT_URL = "http://localhost:6433"   # Qdrant Docker 服务地址
OLLAMA_URL = "http://localhost:11434"  # Ollama 服务地址

def load_paper_db(path):
    """Load paper database from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Paper DB not found at {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def query_vector_db(query_text="Machine Learning", k=3):
    """Test interface to query the built vector database."""
    print(f"\n[Test] Querying Vector DB with: '{query_text}'")
    
    embedding_model_name = "qwen3-embedding:0.6b"
    # 🚀 修改 2: 显式指定 base_url 以防网络问题
    embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=OLLAMA_URL)
    
    COLLECTION_NAME = "paper_knowledge_base"
    
    # 🚀 修改 3: 连接到服务器，而不是本地路径
    client = QdrantClient(url=QDRANT_URL)
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} does not exist.")
        return

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    results = vector_store.similarity_search(query_text, k=k)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Title: {doc.metadata.get('title')}")
        print(f"  ArXiv ID: {doc.metadata.get('arxiv_id')}")
        print(f"  Snippet: {doc.page_content[:150].replace(chr(10), ' ')}...")
        
    client.close()

def build_vector_db():
    # Configuration
    TEST_MODE = False
    TEST_LIMIT = 100
    
    # 1. Load Data
    paper_db_path = config.PAPER_DB_PATH
    print(f"[1/5] Loading paper DB from {paper_db_path}...")
    try:
        paper_db = load_paper_db(paper_db_path)
    except FileNotFoundError:
        print(f"Error: Could not find paper DB at {paper_db_path}")
        return
        
    print(f"      Loaded {len(paper_db)} papers.")

    all_items = list(paper_db.items())
    if TEST_MODE:
        print(f"      [Test Mode] Using first {TEST_LIMIT} papers only.")
        all_items = all_items[:TEST_LIMIT]

    # Checkpoint Setup
    # 🚀 修改 4: Checkpoint 仍然保存在本地文件系统，这没问题
    qdrant_dir = os.path.dirname(config.PAPER_DB_PATH)
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
        doc_list.append(Document(page_content=content, metadata=metadata))
    
    print(f"      Prepared {len(doc_list)} new documents to index.")
    
    if not doc_list:
        print("No new documents to index. All done!")
        return

    # 3. Initialize Embedding & Qdrant
    print("[3/5] Initializing Vector DB (Server Mode)...")
    
    embedding_model_name = "qwen3-embedding:0.6b" 
    embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=OLLAMA_URL)
    
    # 🚀 修改 5: 连接 Docker 服务，移除 path 参数
    print(f"      Connecting to Qdrant Server at: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
    
    COLLECTION_NAME = "paper_knowledge_base"
    
    print("      Probing embedding dimension...")
    try:
        dummy_vec = embeddings.embed_query("test")
        vector_size = len(dummy_vec)
        print(f"      Vector Dimension: {vector_size}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=Distance.COSINE,
            )
        )
        print(f"      Created collection: {COLLECTION_NAME} (Server Mode)")
    else:
        print(f"      Collection {COLLECTION_NAME} already exists.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 4. Indexing Loop
    print(f"[4/5] Indexing {len(doc_list)} vectors...")
    batch_size = 128 
    
    for i in tqdm.tqdm(range(0, len(doc_list), batch_size), desc="Indexing Batches"):
        batch = doc_list[i:i+batch_size]
        try:
            vector_store.add_documents(documents=batch)
            
            for doc in batch:
                indexed_keys.add(doc.metadata['_checkpoint_key'])
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(list(indexed_keys), f)
                
        except Exception as e:
            print(f"Error indexing batch {i}: {e}")
            
    print("Done! Vector DB built successfully.")
    client.close()

if __name__ == "__main__":
    build_vector_db()
    
    # Run a test query
    query_vector_db("large language models reasoning")