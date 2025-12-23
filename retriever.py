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

class Generator:
    def __init__(self):
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def generate(self, query, context):
        if "ignore instructions" in query.lower(): return "I cannot do that."
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            # FIX 3: Updated Model Name to 'command-r-08-2024'
            response = self.client.chat(
                message=prompt,
                model="command-r-08-2024", 
                temperature=0.3
            )
            return response.text
        except Exception as e:
            return f"Generation Error: {str(e)}"