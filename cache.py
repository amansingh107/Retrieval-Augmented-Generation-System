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

class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Cache: Connected to Redis")
        except:
            print("⚠️ Cache: Redis failed. Using local dictionary.")
            self.redis_client = None
            self.local_cache = {}

    def _hash(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, query):
        key = "rag_cache:" + self._hash(query)
        if self.redis_client:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        return self.local_cache.get(key)

    def set(self, query, response, ttl=3600):
        key = "rag_cache:" + self._hash(query)
        val = json.dumps(response)
        if self.redis_client:
            self.redis_client.setex(key, ttl, val)
        else:
            self.local_cache[key] = response