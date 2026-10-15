from core.brain import Brain

def run_tests():
    print("Initializing JARVIS Brain for testing...")
    brain = Brain()
    
    queries = [
        "Search the web for Python programming",
        "Search the web for the latest AI news",
        "Search YouTube for Blender tutorials",
        "Search the web for Roblox mountain seeds"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n{'='*50}\nTEST {i}: {q}\n{'='*50}")
        brain.process_input(q)

if __name__ == "__main__":
    run_tests()
