"""
test_phase6.py
==============
Phase 6 — Search Results Intelligence Unit Tests.

These tests use a mocked LLM and tool registry to verify the Phase 6
pipeline behavior (routing and grounding rules), without needing Ollama
or the internet.

TESTS:
  TEST 1: search_web routing to _search_result_intelligence.
  TEST 2: Duplicate result handling (handled internally by LLM prompt rules).
  TEST 3: "Summarize the first result" triggers LLM Pass 2 with correct history.
  TEST 4: "Open the first result" navigates correctly.
  TEST 5: search_youtube still works (Phase 5 untouched).
  TEST 6: failed search_web still reports honestly.
  TEST 7: normal conversation still works.

Run with:
    python test_phase6.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock


def make_brain(intent_json: str, llm_chat_responses: list):
    """
    Return a Brain with mocked LLM. 
    `llm_chat_responses` is a list of strings the LLM should return on sequential chat() calls.
    (e.g., if we call it for Intelligence then Pass 2, or just Pass 2)
    """
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(side_effect=llm_chat_responses)
    return brain


def mock_execute(registry, tool_name_expected: str, result_json: str):
    def _fake(name, args, confirm_callback=None):
        assert name == tool_name_expected, \
            f"Expected '{tool_name_expected}' but executed '{name}'"
        return result_json
    registry.execute_tool = _fake


class TestPhase6SearchResultsIntelligence(unittest.TestCase):

    # ------------------------------------------------------------------
    # TEST 1 & 2: Search web routing to _search_result_intelligence
    # ------------------------------------------------------------------
    def test_search_web_routes_to_intelligence(self):
        """
        Verify that a successful web search produces a deterministic numbered
        presentation containing the result titles and URLs.

        NOTE: As of Phase 5.3, search_web uses a deterministic presentation layer
        (_present_search_results) rather than an LLM intelligence layer. This test
        was updated to reflect that behavioral change.
        """
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "Python programming"}}',
            llm_chat_responses=[]  # LLM is NOT called for deterministic presentation
        )

        web_result = json.dumps({
            "success": True,
            "tool": "search_web",
            "data": [
                {"title": "Python Docs", "snippet": "Official docs", "url": "docs.python.org"},
                {"title": "W3Schools", "snippet": "Tutorials", "url": "w3schools.com"}
            ],
            "error": None
        })
        mock_execute(brain.registry, "search_web", web_result)

        response = brain.process_input("Search the web for Python programming")
        print(f"\n[TEST 1] Response:\n{response}")

        # Verify the response is deterministic and contains both result titles/URLs
        self.assertIn("Python Docs", response)
        self.assertIn("docs.python.org", response)
        self.assertIn("W3Schools", response)
        self.assertIn("w3schools.com", response)
        # Verify it's a numbered list (deterministic format)
        self.assertIn("1.", response)
        self.assertIn("2.", response)

        # Verify the LLM was NOT called (deterministic presentation, no LLM needed)
        brain.llm.chat.assert_not_called()

    # ------------------------------------------------------------------
    # TEST 3: "Summarize the first result"
    # ------------------------------------------------------------------
    def test_summarize_first_result_grounding(self):
        """
        Verify that Pass 2 handles this and the grounding rule is present.

        Guard F (introduced in Phase 26) checks for an "ACTION EXECUTED:" system
        message in history to confirm a real prior result exists before routing to
        Pass-2 LLM. Without it, Guard F returns a clarification asking the user to
        search first. We inject that marker here to simulate the history state that
        the real pipeline leaves after a successful search_web turn.
        """
        expected_response = "I only have the search snippet for this result, I cannot read the full page."
        brain = make_brain(
            intent_json='{"tool": null, "args": null}',
            llm_chat_responses=[expected_response]
        )

        # Simulate the history state left by a prior successful search_web turn.
        # Brain.process_input inserts base_system_prompt as history[0] only when
        # history is empty. We pre-populate history, so we must seed history[0]
        # with the system prompt ourselves — matching real pipeline layout.
        # Guard F looks for "ACTION EXECUTED:" in a system-role message to confirm
        # a real prior result exists before routing to Pass-2 LLM.
        prior_search_result = json.dumps({
            "success": True,
            "tool": "search_web",
            "data": [
                {"title": "Python Docs", "snippet": "Official docs", "url": "docs.python.org"},
            ],
            "error": None
        })
        brain.history.append({
            "role": "system",
            "content": brain.base_system_prompt   # matches real history[0] layout
        })
        brain.history.append({
            "role": "user",
            "content": "Search the web for Python programming"
        })
        brain.history.append({
            "role": "system",
            "content": f"ACTION EXECUTED: search_web\nRESULT: {prior_search_result}"
        })
        brain.history.append({
            "role": "assistant",
            "content": "1. Python Docs — docs.python.org"
        })

        response = brain.process_input("Summarize the first result")
        print(f"\n[TEST 3] Response: {response}")

        self.assertEqual(response, expected_response)

        args, _ = brain.llm.chat.call_args
        messages = args[0]
        sys_prompt = messages[0]["content"]

        # Verify the new rule 5 was added to the final_system_prompt
        self.assertIn("explicitly state you only have the search snippet", sys_prompt)

    # ------------------------------------------------------------------
    # TEST 4: "Open the first result"
    # ------------------------------------------------------------------
    def test_open_first_result(self):
        """
        Verify correct tool (navigate_browser/open_url) is selected and Phase 5 handles it.
        """
        brain = make_brain(
            intent_json='{"tool": "navigate_browser", "args": {"url": "https://example.com"}}',
            llm_chat_responses=[] # Should not be called, handled deterministically
        )
        
        result = json.dumps({
            "success": True,
            "tool": "navigate_browser",
            "data": "Successfully navigated to https://example.com",
            "error": None
        })
        mock_execute(brain.registry, "navigate_browser", result)

        response = brain.process_input("Open the first result")
        print(f"\n[TEST 4] Response: {response}")

        self.assertIn("navigated the browser", response)
        self.assertIn("https://example.com", response)

    # ------------------------------------------------------------------
    # TEST 5: search_youtube works unchanged
    # ------------------------------------------------------------------
    def test_search_youtube_unchanged(self):
        brain = make_brain(
            intent_json='{"tool": "search_youtube", "args": {"query": "Blender tutorials"}}',
            llm_chat_responses=[]
        )
        yt_result = json.dumps({
            "success": True,
            "tool": "search_youtube",
            "data": "Successfully navigated to YouTube",
            "error": None
        })
        mock_execute(brain.registry, "search_youtube", yt_result)

        response = brain.process_input("Search YouTube for Blender tutorials")
        print(f"\n[TEST 5] Response: {response}")

        self.assertIn("opened YouTube", response)
        self.assertIn("Blender tutorials", response)
        brain.llm.chat.assert_not_called() # Phase 5 deterministic

    # ------------------------------------------------------------------
    # TEST 6: failed search_web
    # ------------------------------------------------------------------
    def test_failed_search_web(self):
        brain = make_brain(
            intent_json='{"tool": "search_web", "args": {"query": "latest AI news"}}',
            llm_chat_responses=[]
        )
        fail_result = json.dumps({
            "success": False,
            "tool": "search_web",
            "data": None,
            "error": "Timeout"
        })
        mock_execute(brain.registry, "search_web", fail_result)

        response = brain.process_input("Search the web for latest AI news")
        print(f"\n[TEST 6] Response: {response}")

        self.assertIn("failed", response)
        self.assertIn("Timeout", response)
        brain.llm.chat.assert_not_called()

    # ------------------------------------------------------------------
    # TEST 7: normal conversation
    # ------------------------------------------------------------------
    def test_normal_conversation(self):
        expected_response = "I am JARVIS."
        brain = make_brain(
            intent_json='{"tool": null, "args": null}',
            llm_chat_responses=[expected_response]
        )

        response = brain.process_input("Who are you?")
        print(f"\n[TEST 7] Response: {response}")

        self.assertEqual(response, expected_response)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — Search Results Intelligence Unit Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase6SearchResultsIntelligence)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
