"""
test_phase5_4.py
================
Phase 5.4 — Performance Optimization Tests.

Verifies that the LLM is entirely bypassed for standard search_web results,
meaning performance goes from O(100s) to O(0.01s).
Also verifies that the new deterministic formatting is correctly applied.

TESTS:
  TEST 1: Normal conversational request still works.
  TEST 2: search_web results remain fully grounded (deterministic).
  TEST 3: search_youtube remains fully grounded.
  TEST 4: 5-result search response does not lose results.
  TEST 5: Empty search results are handled correctly.

Run with:
    python test_phase5_4.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock


def make_brain(intent_json: str, llm_chat_response: str):
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(return_value=llm_chat_response)
    return brain


def mock_execute(registry, tool_name_expected: str, result_json: str):
    def _fake(name, args, confirm_callback=None):
        assert name == tool_name_expected, f"Expected '{tool_name_expected}' but executed '{name}'"
        return result_json
    registry.execute_tool = _fake


class TestPhase5_4Performance(unittest.TestCase):

    # ------------------------------------------------------------------
    # TEST 1: Normal conversational request still works
    # ------------------------------------------------------------------
    def test_conversational_fallback(self):
        expected_response = "The capital of France is Paris."
        brain = make_brain('{"tool": null, "args": null}', expected_response)
        response = brain.process_input("What is the capital of France?")
        self.assertEqual(response, expected_response)

    # ------------------------------------------------------------------
    # TEST 2: search_web results remain fully grounded (deterministic bypass)
    # ------------------------------------------------------------------
    def test_search_web_bypasses_llm(self):
        brain = make_brain('{"tool": "search_web", "args": {"query": "Python"}}', "FAIL: LLM CALLED")
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [
                {"title": "Docs", "snippet": "Official docs", "url": "https://docs.python.org/"}
            ]
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Python")
        
        # Verify LLM was NEVER called for the summary step
        brain.llm.chat.assert_not_called()
        self.assertIn("1. Docs", response)
        self.assertIn("Official docs", response)
        self.assertIn("https://docs.python.org/", response)

    # ------------------------------------------------------------------
    # TEST 3: search_youtube remains fully grounded
    # ------------------------------------------------------------------
    def test_search_youtube_remains_grounded(self):
        brain = make_brain('{"tool": "search_youtube", "args": {"query": "Blender"}}', "FAIL")
        yt_result = json.dumps({"success": True, "tool": "search_youtube", "error": None, "data": "Success"})
        mock_execute(brain.registry, "search_youtube", yt_result)
        response = brain.process_input("Search YouTube for Blender")
        
        brain.llm.chat.assert_not_called()
        self.assertIn("opened YouTube", response)

    # ------------------------------------------------------------------
    # TEST 4: 5-result search response does not lose results
    # ------------------------------------------------------------------
    def test_5_result_search(self):
        brain = make_brain('{"tool": "search_web", "args": {"query": "Test"}}', "FAIL")
        data = []
        for i in range(1, 6):
            data.append({"title": f"Title {i}", "snippet": f"Snippet {i}", "url": f"https://example.com/{i}"})
            
        web_result = json.dumps({"success": True, "tool": "search_web", "error": None, "data": data})
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Test")
        
        brain.llm.chat.assert_not_called()
        for i in range(1, 6):
            self.assertIn(f"{i}. Title {i}", response)
            self.assertIn(f"Snippet {i}", response)

    # ------------------------------------------------------------------
    # TEST 5: Empty search results are handled correctly
    # ------------------------------------------------------------------
    def test_empty_results(self):
        brain = make_brain('{"tool": "search_web", "args": {"query": "empty_query"}}', "FAIL")
        web_result = json.dumps({"success": True, "tool": "search_web", "error": None, "data": []})
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for empty_query")
        
        brain.llm.chat.assert_not_called()
        self.assertIn("found no results", response)

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5.4 — Performance Optimization Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5_4Performance)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
