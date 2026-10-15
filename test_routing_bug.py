import sys, os, json
sys.path.append('.')
from core.brain import Brain
from core.tool_registry import ToolRegistry

b = Brain()
b.history.append({'role': 'user', 'content': 'Search my workspace for MAX_MEMORIES.'})
b.history.append({'role': 'system', 'content': 'ACTION EXECUTED: search_workspace\nRESULT: [{"path": "core/memory.py"}, {"path": "core/brain.py"}]'})

class MockLLM:
    def __init__(self):
        self.model = "mock"
    def chat_json(self, msgs):
        return '{"tool": "get_network_info", "args": {}}'
    def chat(self, msgs):
        return 'Summary response.'

b.llm = MockLLM()
res = b.process_input('Summarize the first result.')
print('FINAL RETURN:', res)
