import redis
import json
import hashlib

class MemoryManager:
    def __init__(self, host='localhost', port=6379):  
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Memory: Connected to Redis")
        except:
            print("⚠️ Memory: Redis failed. Using local memory (RAM).")
            self.redis_client = None
            self.local_cache = {}
            self.chat_history = [] 

    def get_history(self, session_id="user_1"):
        """Retrieve last 5 turns (10 messages)"""
        if self.redis_client:
            history = self.redis_client.lrange(f"hist:{session_id}", 0, -1)
            return [json.loads(h) for h in history]
        return self.chat_history[-10:]

    def add_history(self, session_id, role, content):
        """Add message to history"""
        msg = json.dumps({"role": role, "content": content})
        if self.redis_client:
            key = f"hist:{session_id}"
            self.redis_client.rpush(key, msg)
            self.redis_client.ltrim(key, -10, -1) # Keep last 10 items
        else:
            self.chat_history.append(json.loads(msg))

    def cache_get(self, query):
        """Check for exact query match in cache"""
        key = "cache:" + hashlib.sha256(query.encode()).hexdigest()
        if self.redis_client:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        return self.local_cache.get(key)

    def cache_set(self, query, response):
        """Save answer to cache for 1 hour"""
        key = "cache:" + hashlib.sha256(query.encode()).hexdigest()
        val = json.dumps(response)
        if self.redis_client:
            self.redis_client.setex(key, 3600, val)
        else:
            self.local_cache[key] = response