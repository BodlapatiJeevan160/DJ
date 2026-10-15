import sys
sys.path.append('.')
from core.brain import Brain
import json

b = Brain()
# Mock LLM strictly
class MockLLM:
    def chat_json(self, msgs):
        last_msg = msgs[-1]["content"] if msgs else ""
        if "MAX_MEMORIES" in last_msg:
            return '{"tool": "search_workspace", "args": {"query": "MAX_MEMORIES"}}'
        elif "Summarize" in last_msg:
            return '{"tool": "get_network_info", "args": {}}'
        return '{"tool": null, "args": {}}'
    def chat(self, msgs):
        return "LLM Text Response"

b.llm = MockLLM()

print("--- Turn 1 ---")
res1 = b.process_input("Search my workspace for MAX_MEMORIES.")
print("RES1:", res1)
print("History size:", len(b.history))

print("\n--- Turn 2 ---")
res2 = b.process_input("Summarize the first result.")
print("RES2:", res2)
print("History size:", len(b.history))

for i, m in enumerate(b.history):
    print(f"[{i}] {m['role']}: {m['content'][:50]}...")
