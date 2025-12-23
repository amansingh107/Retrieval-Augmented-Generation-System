import os
# from dotenv import load_dotenv
from src.memory import MemoryManager
from src.tools import Indexer, RetrievalTools
from src.agent import Agent
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

def main():
    # 1. Initialize Components
    mem = MemoryManager()
    idx = Indexer()
    tools = RetrievalTools(idx)
    agent = Agent(tools, mem)

    # 2. Ingest Data (Simulating a document load)
    print("--- Initializing Data ---")
    idx.ingest([
        "The project 'Apollo' deadline is set for March 15th, 2024.",
        "Server passwords must be rotated every 90 days according to security policy.",
        "Contact 'hr@company.com' for leave requests."
    ])
    
    print("\n✅ System Ready! Type 'exit' to quit.\n")

    # 3. Chat Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = agent.run(user_input)
        print("\n")
        print("\n")
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()