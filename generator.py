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

class HybridRetriever:
    def __init__(self, indexer):
        self.idx = indexer
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])

    def search(self, query, top_k=10):
        # Dense Search
        dense = self.idx.collection.query(query_texts=[query], n_results=top_k)
        dense_docs = dense['documents'][0]
        if dense['distances'][0]:
            dense_scores = [1 - x for x in dense['distances'][0]]
        else:
            dense_scores = [0] * len(dense_docs)

        # Sparse Search
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
        if not docs: return []
        try:
            # FIX 1: Updated Model Name
            res = self.co.rerank(
                model="rerank-english-v3.0", 
                query=query, 
                documents=docs, 
                top_n=3
            )
            # FIX 2: Map indices back to original documents
            # The new SDK response returns indices, so we fetch the text from our 'docs' list
            return [docs[x.index] for x in res.results]
        except Exception as e:
            print(f"⚠️ Rerank Error: {e}")
            return docs[:3]