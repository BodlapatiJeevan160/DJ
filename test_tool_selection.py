"""
test_tool_selection.py
======================
Deterministic tests for the Phase 4 tool-selection bug fix.
Does NOT require Ollama to be running — the LLM is mocked.

Tests verify:
1. The Tool Correction guard overrides wrong model decisions.
2. Correct model decisions pass through unchanged.
3. tool_name and tool_args are always coherent before execution.
4. open_browser is NEVER substituted for search_youtube / search_web.

Run with:
    python test_tool_selection.py
"""

import sys
import json
import re
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers — replicate the correction logic from brain.py so we can test it
# in isolation without touching the LLM or browser.
# ---------------------------------------------------------------------------

_YT_PATTERN  = re.compile(r'search(?:\s+on)?\s+youtube\s+for\s+(.+)', re.IGNORECASE)
_WEB_PATTERN = re.compile(r'search(?:\s+the)?\s+web\s+for\s+(.+)', re.IGNORECASE)

def _apply_tool_correction(user_input: str, tool_name: str, tool_args: dict):
    """Mirrors the TOOL CORRECTION block in brain.py."""
    yt_match  = _YT_PATTERN.search(user_input)
    web_match = _WEB_PATTERN.search(user_input)

    if yt_match and tool_name != "search_youtube":
        corrected_query = yt_match.group(1).strip()
        return "search_youtube", {"query": corrected_query}, True   # corrected=True
    elif web_match and tool_name not in ("search_web", "search_youtube"):
        corrected_query = web_match.group(1).strip()
        return "search_web", {"query": corrected_query}, True
    return tool_name, tool_args, False  # corrected=False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestToolCorrectionGuard(unittest.TestCase):

    # --- YouTube search cases ---

    def test_yt_blender_wrong_decision(self):
        """Model returns open_browser for 'Search YouTube for Blender tutorials' -> must be corrected."""
        user_input = "Search YouTube for Blender tutorials"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected,  "Should have been corrected")
        self.assertEqual(tool_name, "search_youtube")
        self.assertEqual(tool_args, {"query": "Blender tutorials"})

    def test_yt_python_wrong_decision(self):
        """Model returns open_browser for 'Search YouTube for Python tutorials' -> must be corrected."""
        user_input = "Search YouTube for Python tutorials"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected,  "Should have been corrected")
        self.assertEqual(tool_name, "search_youtube")
        self.assertEqual(tool_args, {"query": "Python tutorials"})

    def test_yt_ai_wrong_decision(self):
        """Model returns open_browser for 'Search YouTube for AI tutorials' -> must be corrected."""
        user_input = "Search YouTube for AI tutorials"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected,  "Should have been corrected")
        self.assertEqual(tool_name, "search_youtube")
        self.assertEqual(tool_args, {"query": "AI tutorials"})

    def test_yt_correct_decision_passes_through(self):
        """Model correctly returns search_youtube -> no correction needed, args preserved."""
        user_input = "Search YouTube for Python tutorials"
        tool_name, tool_args, corrected = _apply_tool_correction(
            user_input, "search_youtube", {"query": "Python tutorials"}
        )
        self.assertFalse(corrected, "Correct decision should not be overridden")
        self.assertEqual(tool_name, "search_youtube")
        self.assertEqual(tool_args, {"query": "Python tutorials"})

    # --- Web search cases ---

    def test_web_python_wrong_decision(self):
        """Model returns open_browser for 'Search the web for Python programming' -> corrected to search_web."""
        user_input = "Search the web for Python programming"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected,  "Should have been corrected")
        self.assertEqual(tool_name, "search_web")
        self.assertEqual(tool_args, {"query": "Python programming"})

    def test_web_ai_news_wrong_decision(self):
        """Model returns open_browser for 'Search the web for latest AI news' -> corrected to search_web."""
        user_input = "Search the web for latest AI news"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected,  "Should have been corrected")
        self.assertEqual(tool_name, "search_web")
        self.assertEqual(tool_args, {"query": "latest AI news"})

    def test_web_correct_decision_passes_through(self):
        """Model correctly returns search_web -> no correction."""
        user_input = "Search the web for Python programming"
        tool_name, tool_args, corrected = _apply_tool_correction(
            user_input, "search_web", {"query": "Python programming"}
        )
        self.assertFalse(corrected)
        self.assertEqual(tool_name, "search_web")

    # --- Edge cases ---

    def test_open_browser_explicit_request_not_corrected(self):
        """'Open browser' should legitimately use open_browser -- must NOT be corrected."""
        user_input = "Open browser"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertFalse(corrected,  "Explicit open_browser request must not be overridden")
        self.assertEqual(tool_name, "open_browser")

    def test_yt_search_with_on_variant(self):
        """'Search on YouTube for X' variant must also be corrected."""
        user_input = "Search on YouTube for cooking recipes"
        tool_name, tool_args, corrected = _apply_tool_correction(user_input, "open_browser", {})
        self.assertTrue(corrected)
        self.assertEqual(tool_name, "search_youtube")
        self.assertEqual(tool_args, {"query": "cooking recipes"})

    def test_no_correction_for_unrelated_tool(self):
        """A non-search request (e.g., open_application) must not be touched."""
        user_input = "Open Notepad"
        tool_name, tool_args, corrected = _apply_tool_correction(
            user_input, "open_application", {"app_name": "Notepad"}
        )
        self.assertFalse(corrected)
        self.assertEqual(tool_name, "open_application")


# ---------------------------------------------------------------------------
# End-to-end Brain mock test -- verifies the full process_input() pipeline
# without hitting Ollama or the browser.
# ---------------------------------------------------------------------------

class TestBrainIntegration(unittest.TestCase):
    """
    Mocks LLMClient.chat_json to return a WRONG decision (open_browser),
    then verifies that brain.process_input() still executes the corrected tool.
    """

    def _make_brain_with_mocked_llm(self, json_response: str, final_response: str = "Done."):
        """Helper: return a Brain whose LLM is fully mocked."""
        from core.brain import Brain
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=json_response)
        brain.llm.chat      = MagicMock(return_value=final_response)
        return brain

    def _mock_tool_execute(self, registry, expected_tool, expected_args):
        """Patches registry.execute_tool and asserts the right tool is called."""
        def fake_execute(name, args, confirm_callback=None):
            assert name == expected_tool, \
                f"FAIL: Expected tool '{expected_tool}' but got '{name}'"
            assert args == expected_args, \
                f"FAIL: Expected args {expected_args} but got {args}"
            return json.dumps({"success": True, "tool": name, "data": "mock result", "error": None})
        registry.execute_tool = fake_execute

    def test_python_tutorials_corrected_before_execution(self):
        """Even if LLM says open_browser, execution must use search_youtube."""
        brain = self._make_brain_with_mocked_llm(
            json_response='{"tool": "open_browser", "args": {}}'
        )
        self._mock_tool_execute(brain.registry, "search_youtube", {"query": "Python tutorials"})
        brain.process_input("Search YouTube for Python tutorials")

    def test_blender_tutorials_no_correction_needed(self):
        """If LLM already says search_youtube correctly, no correction fires."""
        brain = self._make_brain_with_mocked_llm(
            json_response='{"tool": "search_youtube", "args": {"query": "Blender tutorials"}}'
        )
        self._mock_tool_execute(brain.registry, "search_youtube", {"query": "Blender tutorials"})
        brain.process_input("Search YouTube for Blender tutorials")

    def test_web_python_corrected_before_execution(self):
        """Even if LLM says open_browser, web search must use search_web."""
        brain = self._make_brain_with_mocked_llm(
            json_response='{"tool": "open_browser", "args": {}}'
        )
        self._mock_tool_execute(brain.registry, "search_web", {"query": "Python programming"})
        brain.process_input("Search the web for Python programming")

    def test_web_ai_news_no_correction_needed(self):
        """If LLM already says search_web correctly, no correction fires."""
        brain = self._make_brain_with_mocked_llm(
            json_response='{"tool": "search_web", "args": {"query": "latest AI news"}}'
        )
        self._mock_tool_execute(brain.registry, "search_web", {"query": "latest AI news"})
        brain.process_input("Search the web for latest AI news")

    def test_ai_tutorials_corrected_before_execution(self):
        """Even if LLM says open_browser, AI tutorials must use search_youtube."""
        brain = self._make_brain_with_mocked_llm(
            json_response='{"tool": "open_browser", "args": {}}'
        )
        self._mock_tool_execute(brain.registry, "search_youtube", {"query": "AI tutorials"})
        brain.process_input("Search YouTube for AI tutorials")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Tool-Selection Regression Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestToolCorrectionGuard))
    suite.addTests(loader.loadTestsFromTestCase(TestBrainIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
