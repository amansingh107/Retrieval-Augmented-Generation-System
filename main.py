import os
from dotenv import load_dotenv
from src.memory import MemoryManager
from src.tools import Indexer, RetrievalTools, ContentLoader
from src.agent import Agent

load_dotenv()

def main():
    print("🚀 Initializing Agentic RAG...")
    
    # Initialize
    mem = MemoryManager()
    idx = Indexer()
    loader = ContentLoader()
    tools = RetrievalTools(idx)
    agent = Agent(tools, mem)

    print("\n✅ System Ready!")
    print("Commands:")
    print("  /add <url>       -> Scrape and index a webpage")
    print("  /pdf <path>      -> Read and index a PDF file")
    print("  exit             -> Quit the application\n")

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # --- Handle /add (Web) ---
        if user_input.startswith("/add "):
            url = user_input.split(" ", 1)[1]
            print(f"Trying to add: {url}")
            text_data = loader.load_url(url)
            if text_data:
                idx.ingest(text_data)
            else:
                print("Could not extract text from that URL.")
            continue
            
        # --- Handle /pdf (Files) ---
        if user_input.startswith("/pdf "):
            path = user_input.split(" ", 1)[1]
            if os.path.exists(path):
                text_data = loader.load_pdf(path)
                if text_data:
                    idx.ingest(text_data)
                else:
                    print("Could not extract text from that PDF.")
            else:
                print("❌ File path not found.")
            continue

        # --- Standard Agent Query ---
        response = agent.run(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    main()