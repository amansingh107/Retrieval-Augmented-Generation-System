import json
import hashlib
import string
import numpy as np
import cohere
import chromadb
import redis
import os
from typing import List
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

class Indexer:
    def __init__(self):
        print("⏳ Loading local embedding model (free)...")
        # Uses local CPU/GPU for embeddings to save API costs
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2" 
        )
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("colab_rag_free", embedding_function=self.ef)
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

        if chunks: self.collection.add(documents=chunks, ids=ids)
        self.bm25 = BM25Okapi(tokenized)
        print("✅ Indexing Complete")