"""
test_phase28.py
===============
Phase 28: Restricted Candidate-Tool Routing

Tests cover:
  A. Category mapping
  B. High-confidence routing
  C. Multi-category routing
  D. Ambiguous requests (fallback)
  E. Security validation
  F. Phase 27 regression
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure the project root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.tool_registry import ToolRegistry
from core.brain import Brain, CategoryScorer

class TestPhase28Routing(unittest.TestCase):

    def setUp(self):
        self.brain = Brain()
        self.registry = self.brain.registry
        self.brain.llm = MagicMock()
        self.brain.llm.chat_json.return_value = '{"tool": null, "args": null}'
        self.brain.llm.chat.return_value = "Mock response"
        self.brain.registry.execute_tool = MagicMock(return_value='{"success": true, "data": {}}')
        
        # Patch session save to avoid serialization errors with MagicMock
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_a1_category_mapping(self):
        """Every registered tool belongs to at least one category, and lookups return valid tools."""
        all_categorized_tools = set()
        for cat, tools in ToolRegistry.TOOL_CATEGORIES.items():
            all_categorized_tools.update(tools)
            
            # Lookup returns valid descriptions
            desc = self.registry.get_tool_descriptions({cat})
            self.assertIn("AVAILABLE TOOLS:", desc)
            for tool in tools:
                if tool in self.registry.tools:
                    self.assertIn(f"- {tool}", desc)
                    
        # All tools in registry should be in at least one category
        for name in self.registry.tools:
            self.assertIn(name, all_categorized_tools, f"Tool {name} is missing from all categories")

    def _assert_categories(self, text, expected_cats):
        """Helper to test that the exact set of expected_cats is chosen."""
        self.assertEqual(CategoryScorer.get_routing_categories(text), set(expected_cats))

    def test_b_high_confidence_routing(self):
        """Test that unambiguous requests route to a single category."""
        self._assert_categories("Open Chrome", {"SYSTEM_WM"})
        self._assert_categories("List open windows", {"SYSTEM_WM"})
        self._assert_categories("Search my workspace for something", {"WORKSPACE"})
        self._assert_categories("Remember that my name is Alice", {"MEMORY"})
        self._assert_categories("Set a reminder for 5 minutes", {"NOTIFICATIONS_TIME"})
        self._assert_categories("Turn the volume down", {"MEDIA_AUDIO"})
        self._assert_categories("Take a screenshot", {"VISION"})

    def test_c_multi_category_routing(self):
        """Test overlapping/medium confidence requests and verify correct multi-category merging."""
        # "Open the downloaded PDF" -> open (1 workspace, 1 browser, 1 system_wm), downloaded (3 workspace)
        # WORKSPACE=4, BROWSER=1, SYSTEM_WM=1; lead 4-1=3 > 2 -> HIGH -> WORKSPACE
        scores = CategoryScorer.score_categories("Open the downloaded PDF")
        self.assertEqual(scores["WORKSPACE"], 4)   # open(1) + downloaded(3)
        self.assertEqual(scores["BROWSER"], 1)     # open(1)
        self.assertEqual(scores["SYSTEM_WM"], 1)   # open(1)
        self.assertEqual(CategoryScorer.get_routing_categories("Open the downloaded PDF"), {"WORKSPACE"})
        
        # "Open Chrome" -> SYSTEM_WM has 5pt phrase + open(1) + chrome(1) = 7; BROWSER = open(1) = 1
        # Lead 7-1=6 > 2 -> HIGH -> SYSTEM_WM
        self.assertEqual(CategoryScorer.get_routing_categories("Open Chrome"), {"SYSTEM_WM"})
        
        # "Search Google and find the file" -> BROWSER: search(1)+Google(3)=4; WORKSPACE: search(1)+find(1)+file(3)=5
        # WORKSPACE=5, BROWSER=4; 5-4=1 <= 2 -> MEDIUM -> merge both
        result = CategoryScorer.get_routing_categories("Search Google and find the file")
        self.assertIn("BROWSER", result)
        self.assertIn("WORKSPACE", result)

    def test_d_ambiguous_requests(self):
        """Test fallback behavior for low or very low confidence."""
        # "Make it happen" -> no keywords -> VERY LOW -> ALL (empty set means ALL)
        self._assert_categories("Make it happen", set())
        
        # "Do the thing we discussed" -> no keywords -> VERY LOW
        self._assert_categories("Do the thing we discussed", set())
        
        # "Help me with this" -> no keywords -> VERY LOW
        self._assert_categories("Help me with this", set())
        
        # LOW confidence: 1 or 2 points total.
        # "look at this" -> look(1 VISION) -> best score = 1
        self._assert_categories("look at this", {"GENERAL"})
        
        # "set this up" -> set(1 MEDIA, 1 TIME) -> best score = 1
        self._assert_categories("set this up", {"GENERAL"})

    def test_e_security(self):
        """Ensure candidate routing does not directly execute tools, and ToolRegistry remains the authority."""
        # Fake LLM returning a dangerous tool from a restricted subset
        self.brain.llm.chat_json.return_value = '{"tool": "close_window", "args": {"hwnd": 123}}'
        
        # We process input (which should route via LLM)
        self.brain.process_input("Kill the window")
        
        # execute_tool must still be called so ToolRegistry can handle the danger
        self.brain.registry.execute_tool.assert_called()
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "close_window")
        self.assertIsNotNone(kwargs.get("confirm_callback"))

    def test_f_phase27_regression(self):
        """Ensure conversational pre-classifier still bypasses Pass-1 and session persistence works."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        
        # Obvious statement -> skip LLM Pass-1
        self.brain.process_input("The weather is nice today.")
        self.brain.llm.chat_json.assert_not_called()
        self.brain.registry.execute_tool.assert_not_called()
        
        # Historical recall (NEBULA-style)
        self.brain.process_input("What was the unique session marker I gave you?")
        self.brain.llm.chat_json.assert_not_called()
        self.brain.registry.execute_tool.assert_not_called()

    def test_g_routing_bug_repro(self):
        """Ensure deterministic routing handles 'Show me my open windows' etc., and JSON parse errors fail-fast."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        
        # 1-4. Deterministic fast paths for window listing
        for text in [
            "Show me my open windows",
            "List my open windows",
            "What windows are open?",
            "Show open windows"
        ]:
            self.brain.registry.execute_tool.reset_mock()
            self.brain.process_input(text)
            self.brain.registry.execute_tool.assert_called()
            args, kwargs = self.brain.registry.execute_tool.call_args
            self.assertEqual(kwargs.get("name"), "list_windows", f"Failed on: {text}")
            
        # 5. Open Chrome -> open_application
        self.brain.registry.execute_tool.reset_mock()
        self.brain.process_input("Open Chrome")
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "open_application")
        
        # 6. Open the downloaded PDF -> no fast path, goes to LLM
        self.brain.llm.chat_json.reset_mock()
        self.brain.process_input("Open the downloaded PDF")
        self.brain.llm.chat_json.assert_called()
        
        # 7-8. Malformed JSON recovery
        self.brain.llm.chat_json.return_value = '{"list_windows": [{"title": "fake"'
        self.brain.registry.execute_tool.reset_mock()
        # "Are there any hidden windows?" -> goes to LLM -> LLM returns malformed JSON -> should recover list_windows
        self.brain.process_input("Are there any hidden windows?")
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "list_windows")
        
        # Unrecoverable malformed JSON -> no unrestricted LLM generation
        self.brain.llm.chat_json.return_value = '{"fake_tool": "hello"}'
        self.brain.llm.chat.reset_mock()
        result = self.brain.process_input("Will this random thing work?")
        # Should return a safe deterministic fallback message
        self.assertIn("encountered an internal error", result.get("response", ""))
        # Final LLM chat should NOT be called since it failed fast
        self.brain.llm.chat.assert_not_called()

if __name__ == "__main__":
    unittest.main(verbosity=2)
