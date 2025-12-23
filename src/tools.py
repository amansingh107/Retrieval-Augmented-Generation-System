import os
import string
import numpy as np
import cohere
import chromadb
from typing import List
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS

class Indexer:
    def __init__(self, persist_dir="./chroma_db"):
        print("⏳ Loading embedding model...")
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Use PersistentClient to save data to disk
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="agentic_rag", 
            embedding_function=self.ef
        )
        self.bm25 = None
        self.doc_map = {}

    def ingest(self, texts: List[str]):
        print(f"📥 Ingesting {len(texts)} documents...")
        chunks, ids, tokenized = [], [], []
        
        for idx, text in enumerate(texts):
            doc_id = f"doc_{idx}"
            chunks.append(text)
            ids.append(doc_id)
            tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
            tokenized.append(tokens)
            self.doc_map[idx] = text

        if chunks: 
            # Check if IDs exist to avoid duplicates (simplified)
            try:
                self.collection.add(documents=chunks, ids=ids)
            except:
                pass 
        
        self.bm25 = BM25Okapi(tokenized)
        print("✅ Indexing Complete")

class RetrievalTools:
    def __init__(self, indexer):
        self.idx = indexer
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.ddgs = DDGS()

    def web_search(self, query, max_results=3):
        print(f"🌐 Searching Web for: {query}...")
        try:
            results = self.ddgs.text(query, max_results=max_results)
            return [f"Source: {r['title']}\nSnippet: {r['body']}" for r in results]
        except Exception as e:
            print(f"Web Search Error: {e}")
            return []

    def vector_search(self, query, top_k=5):
        # Dense
        dense = self.idx.collection.query(query_texts=[query], n_results=top_k)
        dense_docs = dense['documents'][0]
        if dense['distances'][0]:
            dense_scores = [1 - x for x in dense['distances'][0]]
        else:
            dense_scores = [0] * len(dense_docs)

        # Sparse
        if not self.idx.bm25: return dense_docs # Fallback if no BM25 built
        
        tokens = query.lower().split()
        bm25_scores = self.idx.bm25.get_scores(tokens)
        top_n = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_docs = [self.idx.doc_map[i] for i in top_n]
        sparse_scores = [bm25_scores[i] for i in top_n]

        # Fusion
        def normalize(lst):
            if not lst: return []
            mn, mx = min(lst), max(lst)
            if mx == mn: return [1.0] * len(lst)
            return [(x - mn)/(mx - mn) for x in lst]

        d_norm = normalize(dense_scores)
        s_norm = normalize(sparse_scores)

        scores = {}
        for d, s in zip(dense_docs, d_norm): scores[d] = scores.get(d, 0) + (s * 0.7)
        for d, s in zip(sparse_docs, s_norm): scores[d] = scores.get(d, 0) + (s * 0.3)

        return sorted(scores, key=scores.get, reverse=True)[:top_k]

    def rerank(self, query, docs):
        if not docs: return [], []
        try:
            res = self.co.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=3)
            return [docs[x.index] for x in res.results], [x.relevance_score for x in res.results]
        except:
            return docs[:3], [0.5]*3