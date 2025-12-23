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
from cache import CacheManager
from indexer import Indexer
from retriever import Generator
from generator import HybridRetriever

# 1. Initialize Pipeline
cache = CacheManager()
indexer = Indexer()

# 2. Add Dummy Data
sample_data = [
    "The 'Project Alpha' return policy lasts 30 days. Contact support@alpha.com.",
    "Project Alpha supports 4K video rendering on the Enterprise plan only.",
    "Standard users are limited to 1080p export resolution.",
    "To reset your password, click 'Forgot Password' on the login screen."
]
indexer.ingest(sample_data)

retriever = HybridRetriever(indexer)
generator = Generator()

# 3. Main Query Function
def ask_rag(query):
    print(f"\n❓ Query: {query}")
    if hit := cache.get(query): return f"⚡ CACHED: {hit}"
    
    candidates = retriever.search(query)
    final_docs = retriever.rerank(query, candidates)
    
    if not final_docs: return "No info found."
    
    answer = generator.generate(query, "\n".join(final_docs))
    cache.set(query, answer)
    return f"🤖 LLM: {answer}"

# --- TEST ---
print(ask_rag("What is the return policy?"))