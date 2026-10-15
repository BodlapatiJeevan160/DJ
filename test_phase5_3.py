"""
test_phase5_3.py
================
Phase 5.3 — Grounded Result Presentation Layer Tests.

These tests use a mocked LLM and tool registry to verify the Phase 5.3
pipeline behavior (routing, domain extraction, deduplication, formatting)
without needing Ollama or the internet.

TESTS:
  TEST 1: Multiple results from different domains.
  TEST 2: Multiple results from the same domain.
  TEST 3: Missing/empty snippets.
  TEST 4: Duplicate URLs.
  TEST 5: Zero-result search.

Run with:
    python test_phase5_3.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock


def make_brain(intent_json: str, llm_chat_responses: list):
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(side_effect=llm_chat_responses)
    return brain


def mock_execute(registry, tool_name_expected: str, result_json: str):
    def _fake(name, args, confirm_callback=None):
        assert name == tool_name_expected, f"Expected '{tool_name_expected}' but executed '{name}'"
        return result_json
    registry.execute_tool = _fake


class TestPhase5_3Presentation(unittest.TestCase):

    # ------------------------------------------------------------------
    # TEST 1: Multiple results from different domains
    # ------------------------------------------------------------------
    def test_different_domains(self):
        """Verify prompt contains different domains correctly extracted."""
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "Python"}}',
            llm_chat_responses=["Mock Summary"]
        )
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [
                {"title": "Docs", "snippet": "A", "url": "https://docs.python.org/3/"},
                {"title": "Tutorial", "snippet": "B", "url": "https://www.w3schools.com/python/"}
            ]
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Python")
        
        # Verify the deterministic formatting contains the titles and domains
        self.assertIn("1. Docs", response)
        self.assertIn("https://docs.python.org/3/", response)
        self.assertIn("2. Tutorial", response)
        self.assertIn("https://www.w3schools.com/python/", response)

    # ------------------------------------------------------------------
    # TEST 2: Multiple results from the same domain
    # ------------------------------------------------------------------
    def test_same_domain_grouping_rule(self):
        """Verify output includes both results from the same domain."""
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "Python"}}',
            llm_chat_responses=["Mock Summary"]
        )
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [
                {"title": "Docs 1", "snippet": "A", "url": "https://docs.python.org/1/"},
                {"title": "Docs 2", "snippet": "B", "url": "https://docs.python.org/2/"}
            ]
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Python")
        
        self.assertIn("1. Docs 1", response)
        self.assertIn("2. Docs 2", response)

    # ------------------------------------------------------------------
    # TEST 3: Missing/empty snippets
    # ------------------------------------------------------------------
    def test_missing_empty_snippets(self):
        """Verify prompt handles missing snippets without failing."""
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "Python"}}',
            llm_chat_responses=["Mock Summary"]
        )
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [
                {"title": "No Snippet 1", "snippet": "", "url": "https://example.com/1"},
                {"title": "No Snippet 2", "url": "https://example.com/2"} # Missing key entirely
            ]
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Python")
        
        self.assertEqual(response.count("(No description available)"), 2)

    # ------------------------------------------------------------------
    # TEST 4: Duplicate URLs
    # ------------------------------------------------------------------
    def test_duplicate_urls(self):
        """Verify duplicate URLs are removed before hitting the LLM."""
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "Python"}}',
            llm_chat_responses=["Mock Summary"]
        )
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": [
                {"title": "A", "snippet": "A", "url": "https://example.com"},
                {"title": "A duplicate", "snippet": "A", "url": "https://example.com"}
            ]
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for Python")
        
        # Should only be one RESULT
        self.assertIn("1. A", response)
        self.assertNotIn("2.", response)

    # ------------------------------------------------------------------
    # TEST 5: Zero-result search
    # ------------------------------------------------------------------
    def test_zero_results(self):
        """Verify empty data list bypasses LLM and returns deterministically."""
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "asdfghjkl"}}',
            llm_chat_responses=[] # Should not be called
        )
        web_result = json.dumps({
            "success": True, "tool": "search_web", "error": None,
            "data": []
        })
        mock_execute(brain.registry, "search_web", web_result)
        response = brain.process_input("Search the web for asdfghjkl")
        
        self.assertIn("found no results", response)
        brain.llm.chat.assert_not_called()

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5.3 — Grounded Presentation Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5_3Presentation)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
