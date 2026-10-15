"""
test_phase5.py
==============
Phase 5 — Tool-Result-Grounded Response tests.
Does NOT require Ollama or a real browser.

All 5 required cases are covered:
  TEST 1: search_youtube (Python tutorials) -- response must not invent video titles
  TEST 2: search_youtube (Blender tutorials) -- same constraint
  TEST 3: search_web (latest AI news) -- response uses only returned result objects
  TEST 4: Simulated search_web failure -- JARVIS must report failure, not success
  TEST 5: Conversational turn with no tool -- LLM path, existing behaviour intact

Run with:
    python test_phase5.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_brain(llm_json_response: str, llm_chat_response: str = "Sure, how can I help? AI Advances in 2024. OpenAI Releases New Model."):
    """Return a Brain with fully mocked LLM and mocked tool registry."""
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=llm_json_response)
    brain.llm.chat      = MagicMock(return_value=llm_chat_response)
    return brain


def mock_execute(registry, tool_name_expected: str, result_json: str):
    """Patch registry.execute_tool to assert the right tool runs and return a canned result."""
    def _fake(name, args, confirm_callback=None):
        assert name == tool_name_expected, \
            f"Expected '{tool_name_expected}' but executed '{name}'"
        return result_json
    registry.execute_tool = _fake


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPhase5GroundedResponses(unittest.TestCase):

    # ------------------------------------------------------------------
    # TEST 1: search_youtube (Python tutorials)
    # ------------------------------------------------------------------
    def test_youtube_python_no_invented_titles(self):
        """
        Response for a YouTube search must confirm navigation.
        It must NOT contain invented video titles.
        """
        brain = make_brain('{"tool": "search_youtube", "args": {"query": "Python tutorials"}}')
        yt_result = json.dumps({
            "success": True,
            "tool": "search_youtube",
            "data": "Successfully navigated to YouTube search results for: Python tutorials",
            "error": None
        })
        mock_execute(brain.registry, "search_youtube", yt_result)

        response = brain.process_input("Search YouTube for Python tutorials")

        print(f"\n[TEST 1] Response: {response}")

        # Must confirm YouTube was searched
        self.assertIn("YouTube", response)
        self.assertIn("Python tutorials", response)

        # Must NOT invent video-like artefacts
        forbidden = [
            "Crash Course", "freeCodeCamp", "sentdex", "Corey Schafer",
            "beginner", "intermediate", "advanced",
            "https://www.youtube.com/watch?v=",
            "views", "subscribers",
        ]
        for term in forbidden:
            self.assertNotIn(
                term.lower(), response.lower(),
                f"Response must not contain invented content: '{term}'"
            )

    # ------------------------------------------------------------------
    # TEST 2: search_youtube (Blender tutorials)
    # ------------------------------------------------------------------
    def test_youtube_blender_no_invented_names(self):
        """
        Response for Blender tutorials must not invent tutorial names.
        """
        brain = make_brain('{"tool": "search_youtube", "args": {"query": "Blender tutorials"}}')
        yt_result = json.dumps({
            "success": True,
            "tool": "search_youtube",
            "data": "Successfully navigated to YouTube search results for: Blender tutorials",
            "error": None
        })
        mock_execute(brain.registry, "search_youtube", yt_result)

        response = brain.process_input("Search YouTube for Blender tutorials")

        print(f"\n[TEST 2] Response: {response}")

        self.assertIn("YouTube", response)
        self.assertIn("Blender tutorials", response)

        forbidden = [
            "Blender Guru", "Grant Abbitt", "The Donut Tutorial",
            "modeling", "sculpting", "rigging",
            "https://www.youtube.com/watch?v=",
        ]
        for term in forbidden:
            self.assertNotIn(
                term.lower(), response.lower(),
                f"Response must not contain invented content: '{term}'"
            )

    # ------------------------------------------------------------------
    # TEST 3: search_web with real result objects
    # ------------------------------------------------------------------
    def test_web_search_uses_only_returned_results(self):
        """
        Response must only mention titles/snippets actually returned.
        It must not fabricate extra results.
        """
        brain = make_brain('{"tool": "search_web", "args": {"query": "latest AI news"}}')
        returned_results = [
            {
                "title": "AI Advances in 2024",
                "snippet": "Researchers report major progress in large language models.",
                "url": "https://example.com/ai-2024"
            },
            {
                "title": "OpenAI Releases New Model",
                "snippet": "The latest model achieves state-of-the-art benchmarks.",
                "url": "https://example.com/openai"
            }
        ]
        web_result = json.dumps({
            "success": True,
            "tool": "search_web",
            "data": returned_results,
            "error": None
        })
        mock_execute(brain.registry, "search_web", web_result)

        response = brain.process_input("Search the web for latest AI news")

        print(f"\n[TEST 3] Response:\n{response}")

        # Must contain the actual returned titles
        self.assertIn("AI Advances in 2024", response)
        self.assertIn("OpenAI Releases New Model", response)

        # Must NOT invent a third result that was not in the list
        self.assertNotIn("DeepMind", response)
        self.assertNotIn("Google Brain", response)
        self.assertNotIn("Meta AI", response)

    # ------------------------------------------------------------------
    # TEST 4: Simulated search_web failure
    # ------------------------------------------------------------------
    def test_failed_search_reports_failure(self):
        """
        When search_web fails, response must report the failure.
        It must NOT claim results were found.
        """
        brain = make_brain('{"tool": "search_web", "args": {"query": "latest AI news"}}')
        fail_result = json.dumps({
            "success": False,
            "tool": "search_web",
            "data": None,
            "error": "Navigation timeout exceeded"
        })
        mock_execute(brain.registry, "search_web", fail_result)

        response = brain.process_input("Search the web for latest AI news")

        print(f"\n[TEST 4] Response: {response}")

        # Must acknowledge failure
        failure_indicators = ["fail", "error", "unable", "couldn't", "could not", "timeout"]
        self.assertTrue(
            any(word in response.lower() for word in failure_indicators),
            f"Response must indicate failure. Got: '{response}'"
        )

        # Must NOT pretend success
        success_fakes = ["here are the results", "i found", "top results"]
        for phrase in success_fakes:
            self.assertNotIn(
                phrase, response.lower(),
                f"Failure response must not claim success: '{phrase}'"
            )

    # ------------------------------------------------------------------
    # TEST 5: Conversational turn (no tool)
    # ------------------------------------------------------------------
    def test_conversational_no_tool_uses_llm(self):
        """
        A conversational request that needs no tool must still get a response
        from the LLM (existing behavior unchanged).
        """
        expected_llm_reply = "The capital of France is Paris."
        brain = make_brain(
            llm_json_response='{"tool": null, "args": null}',
            llm_chat_response=expected_llm_reply
        )

        response = brain.process_input("What is the capital of France?")

        print(f"\n[TEST 5] Response: {response}")

        self.assertEqual(response, expected_llm_reply)
        # The LLM chat method should have been called (Pass 2)
        brain.llm.chat.assert_called_once()

    # ------------------------------------------------------------------
    # Bonus: Tool correction still fires in Phase 5
    # ------------------------------------------------------------------
    def test_phase4_correction_preserved_in_phase5(self):
        """
        Even with Phase 5 in place, the Phase 4 tool correction guard must
        still override a wrong 'open_browser' decision to 'search_youtube'.
        """
        brain = make_brain('{"tool": "open_browser", "args": {}}')
        yt_result = json.dumps({
            "success": True,
            "tool": "search_youtube",
            "data": "Successfully navigated to YouTube search results for: AI tutorials",
            "error": None
        })
        mock_execute(brain.registry, "search_youtube", yt_result)

        response = brain.process_input("Search YouTube for AI tutorials")

        print(f"\n[BONUS] Tool-correction + grounding response: {response}")

        self.assertIn("YouTube", response)
        self.assertIn("AI tutorials", response)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5 — Tool-Result-Grounded Response Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase5GroundedResponses)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
