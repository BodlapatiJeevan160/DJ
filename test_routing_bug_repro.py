import unittest
import json
from unittest.mock import patch, MagicMock
from core.brain import Brain
from core.tool_registry import ToolRegistry

class TestPhase26RoutingBug(unittest.TestCase):
    def setUp(self):
        self.brain = Brain()
        
    def test_routing_bug_reproduction(self):
        # 1. Search my workspace for MAX_MEMORIES
        # Mock LLM for Pass 1 to return search_workspace
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "search_workspace", "args": {"query": "MAX_MEMORIES"}}')
        # Mock LLM for Pass 2 to return a dummy response
        self.brain.llm.chat = MagicMock(return_value='Search results...')
        
        # Mock registry to return a fake result
        self.brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, 
            "data": {"results": [{"file": "f1.py", "match": "MAX_MEMORIES = 5"}]}
        }))
        
        response1 = self.brain.process_input("Search my workspace for MAX_MEMORIES.")
        self.brain.registry.execute_tool.assert_called_with(
            name="search_workspace", args={"query": "MAX_MEMORIES", "search_type": "content"}, confirm_callback=self.brain._ask_confirmation
        )
        self.brain.registry.execute_tool.reset_mock()
        
        # 2. Summarize the first result.
        # Mock LLM to INCORRECTLY return get_network_info
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        self.brain.llm.chat = MagicMock(return_value='Summary of the first result is MAX_MEMORIES=5.')
        
        response2 = self.brain.process_input("Summarize the first result.")
        
        # get_network_info SHOULD NOT be executed
        self.brain.registry.execute_tool.assert_not_called()
        self.assertIn('Summary of the first result', response2)

if __name__ == '__main__':
    unittest.main()
