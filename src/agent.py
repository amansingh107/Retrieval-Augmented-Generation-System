import os
import cohere
from .tools import RetrievalTools
from .memory import MemoryManager

class Agent:
    def __init__(self, tools: RetrievalTools, memory: MemoryManager):
        self.tools = tools
        self.memory = memory
        self.client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def _route_query(self, query):
        q_lower = query.lower()
        if any(w in q_lower for w in ["latest", "current", "today", "news", "price"]):
            return "web"
        if any(w in q_lower for w in ["hi", "hello", "thanks", "help", "bye"]):
            return "chat"
        return "rag"

    def run(self, query, session_id="user_1"):
        print(f"\n🤖 Agent received: {query}")
        
        # 1. Check Cache
        if hit := self.memory.cache_get(query):
            return f"⚡ CACHED: {hit}"

        # 2. Router
        route = self._route_query(query)
        print(f"🧭 Routing to: {route.upper()}")

        context = ""
        
        # 3. Execution
        if route == "rag":
            docs = self.tools.vector_search(query)
            reranked_docs, scores = self.tools.rerank(query, docs)
            
            # Fallback logic
            if not scores or max(scores) < 0.1:
                print("⚠️ Low confidence. Switching to Web Search...")
                web_docs = self.tools.web_search(query)
                context = "\n".join(web_docs)
            else:
                context = "\n".join(reranked_docs)

        elif route == "web":
            web_docs = self.tools.web_search(query)
            context = "\n".join(web_docs)

        # 4. Generate
        history = self.memory.get_history(session_id)
        chat_history = []
        for msg in history:
            chat_history.append({"role": "USER" if msg['role'] == "user" else "CHATBOT", "message": msg['content']})

        try:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
            response = self.client.chat(
                message=prompt,
                model="command-r-08-2024",
                chat_history=chat_history,
                temperature=0.3
            )
            answer = response.text
            
            # 5. Save
            self.memory.add_history(session_id, "user", query)
            self.memory.add_history(session_id, "ai", answer)
            self.memory.cache_set(query, answer)
            
            return answer
        except Exception as e:
            return f"Error: {e}"