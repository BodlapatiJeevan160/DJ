"""
test_browser_search_fastpath.py
================================
Tests for Fast-path K: general browser/web-search phrases (brain.py).

Verifies:
  1. "search for X"           -> search_web, no Pass-1 LLM
  2. "look up X"              -> search_web, no Pass-1 LLM
  3. "find X online"          -> search_web, no Pass-1 LLM
  4. "google X" (as verb)     -> search_web, no Pass-1 LLM
  5. "tell me the latest X"   -> search_web, no Pass-1 LLM
  6. Existing _WEB_PATTERN / _YT_PATTERN fast-paths unchanged
  7. Conversational questions NOT intercepted
  8. "open google and tell me X" NOT silently converted to single search
  9. ToolRegistry is execution authority (execute_tool called, not raw skill)
 10. workspace-search ("search for workspace X") NOT intercepted
"""

import pytest
from unittest.mock import MagicMock, patch

from core.brain import (
    Brain,
    _BROWSER_SEARCH_PATTERN,
    _WEB_PATTERN,
    _YT_PATTERN,
)
from core.session import SessionStore


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #

@pytest.fixture
def brain():
    """Isolated Brain with mocked LLM and session persistence."""
    with patch.object(SessionStore, "save"):
        b = Brain()
        b.llm.chat_json = MagicMock(return_value='"{"tool": null, "args": null}"')
        b.llm.chat = MagicMock(return_value="Here are the search results.")
        yield b


def _run(brain, user_input):
    """Run process_input with execute_tool mocked. Returns the mock."""
    fake_result = (
        '"{"success": true, "tool": "search_web", '
        '"data": [{"title": "t", "snippet": "s", "url": "https://example.com"}], '
        '"error": null}"'
    )
    with patch.object(brain.registry, "execute_tool", return_value=fake_result) as mock_exec:
        brain.process_input(user_input)
    return mock_exec


# --------------------------------------------------------------------------- #
# Section 1: Pattern unit tests
# --------------------------------------------------------------------------- #

class TestBrowserSearchPatternUnit:
    """Direct regex tests -- no Brain, no LLM."""

    def test_search_for_captures_query(self):
        m = _BROWSER_SEARCH_PATTERN.match("search for latest AI models")
        assert m is not None
        query = next(g for g in m.groups() if g is not None)
        assert query.strip() == "latest AI models"

    def test_search_for_case_insensitive(self):
        assert _BROWSER_SEARCH_PATTERN.match("Search For Python tutorials") is not None

    def test_search_for_with_please(self):
        assert _BROWSER_SEARCH_PATTERN.match("please search for best laptops 2024") is not None

    def test_search_for_excludes_web(self):
        assert _BROWSER_SEARCH_PATTERN.match("search for web results") is None

    def test_search_for_excludes_youtube(self):
        assert _BROWSER_SEARCH_PATTERN.match("search for youtube videos") is None

    def test_search_for_excludes_workspace(self):
        assert _BROWSER_SEARCH_PATTERN.match("search for workspace README") is None

    def test_search_for_excludes_project(self):
        assert _BROWSER_SEARCH_PATTERN.match("search for project structure") is None

    def test_search_for_excludes_codebase(self):
        assert _BROWSER_SEARCH_PATTERN.match("search for codebase issues") is None

    def test_look_up_captures_query(self):
        m = _BROWSER_SEARCH_PATTERN.match("look up latest AI models")
        assert m is not None
        query = next(g for g in m.groups() if g is not None)
        assert query.strip() == "latest AI models"

    def test_look_up_case_insensitive(self):
        assert _BROWSER_SEARCH_PATTERN.match("Look Up Python documentation") is not None

    def test_look_up_with_please(self):
        assert _BROWSER_SEARCH_PATTERN.match("please look up the weather") is not None

    def test_find_online_captures_query(self):
        m = _BROWSER_SEARCH_PATTERN.match("find latest AI models online")
        assert m is not None
        query = next(g for g in m.groups() if g is not None)
        assert query.strip() == "latest AI models"

    def test_find_online_with_please(self):
        assert _BROWSER_SEARCH_PATTERN.match("please find best recipes online") is not None

    def test_find_without_online_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("find README") is None

    def test_find_and_read_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("find README and read it") is None

    def test_google_verb_captures_query(self):
        m = _BROWSER_SEARCH_PATTERN.match("google latest AI models")
        assert m is not None
        query = next(g for g in m.groups() if g is not None)
        assert query.strip() == "latest AI models"

    def test_google_verb_case_insensitive(self):
        assert _BROWSER_SEARCH_PATTERN.match("Google Python tutorials") is not None

    def test_tell_me_latest_captures_query(self):
        m = _BROWSER_SEARCH_PATTERN.match("tell me the latest AI models")
        assert m is not None
        query = next(g for g in m.groups() if g is not None)
        assert "AI models" in query

    def test_tell_me_current_matches(self):
        assert _BROWSER_SEARCH_PATTERN.match("tell me current news about AI") is not None

    def test_tell_me_recent_matches(self):
        assert _BROWSER_SEARCH_PATTERN.match("tell me recent breakthroughs in science") is not None

    def test_tell_me_newest_matches(self):
        assert _BROWSER_SEARCH_PATTERN.match("tell me newest smartphones") is not None

    def test_tell_me_without_time_marker_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("tell me your name") is None

    def test_tell_me_about_X_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("tell me about Python") is None

    def test_open_google_and_X_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("open google and tell me latest AI models") is None

    def test_open_X_and_search_Y_does_not_match(self):
        assert _BROWSER_SEARCH_PATTERN.match("open chrome and search for AI news") is None

    def test_web_pattern_still_matches_search_the_web_for(self):
        assert _WEB_PATTERN.search("search the web for latest AI models") is not None

    def test_yt_pattern_still_matches_search_youtube_for(self):
        assert _YT_PATTERN.search("search youtube for Python tutorials") is not None


# --------------------------------------------------------------------------- #
# Section 2: Brain integration tests
# --------------------------------------------------------------------------- #

class TestBrowserSearchFastPathIntegration:
    """Integration tests: Brain.process_input + mocked ToolRegistry."""

    def test_search_for_calls_search_web(self, brain):
        mock_exec = _run(brain, "search for latest AI models")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert args.get("query") == "latest AI models"
        brain.llm.chat_json.assert_not_called()

    def test_look_up_calls_search_web(self, brain):
        mock_exec = _run(brain, "look up latest AI models")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert args.get("query") == "latest AI models"
        brain.llm.chat_json.assert_not_called()

    def test_find_online_calls_search_web(self, brain):
        mock_exec = _run(brain, "find latest AI models online")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert args.get("query") == "latest AI models"
        brain.llm.chat_json.assert_not_called()

    def test_google_verb_calls_search_web(self, brain):
        mock_exec = _run(brain, "google latest AI models")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert "AI models" in args.get("query", "")
        brain.llm.chat_json.assert_not_called()

    def test_tell_me_latest_calls_search_web(self, brain):
        mock_exec = _run(brain, "tell me the latest AI models")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert "AI models" in args.get("query", "")
        brain.llm.chat_json.assert_not_called()

    def test_search_the_web_for_still_works(self, brain):
        mock_exec = _run(brain, "search the web for latest AI models")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"
        assert args.get("query") == "latest AI models"
        brain.llm.chat_json.assert_not_called()

    def test_search_youtube_for_still_works(self, brain):
        mock_exec = _run(brain, "search on youtube for Python tutorials")
        assert mock_exec.called
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_youtube"
        assert args.get("query") == "Python tutorials"
        brain.llm.chat_json.assert_not_called()

    def test_open_google_and_tell_me_not_fast_pathed(self, brain):
        """Compound request must reach Pass-1, not be silently converted."""
        with patch.object(brain.registry, "execute_tool",
                          return_value='{"success": false, "error": "noop", "data": null}'):
            brain.process_input("open google and tell me latest AI models")
        brain.llm.chat_json.assert_called()

    def test_what_is_python_not_intercepted(self, brain):
        with patch.object(brain.registry, "execute_tool") as mock_exec:
            brain.process_input("what is Python?")
        mock_exec.assert_not_called()

    def test_how_are_you_not_intercepted(self, brain):
        with patch.object(brain.registry, "execute_tool") as mock_exec:
            brain.process_input("Hello JARVIS, how are you?")
        mock_exec.assert_not_called()

    def test_tell_me_your_name_not_intercepted(self, brain):
        with patch.object(brain.registry, "execute_tool") as mock_exec:
            brain.process_input("tell me your name")
        mock_exec.assert_not_called()

    def test_toolregistry_is_execution_authority(self, brain):
        mock_exec = _run(brain, "search for best Python books")
        mock_exec.assert_called_once()
        name, args, *_ = mock_exec.call_args[0]
        assert name == "search_web"

    def test_fast_path_does_not_bypass_toolregistry(self, brain):
        with patch("skills.web_browser.search_web") as raw_skill:
            with patch.object(brain.registry, "execute_tool",
                              return_value='{"success": true, "data": [], "tool": "search_web"}'):
                brain.process_input("look up best Python IDE")
        raw_skill.assert_not_called()

    def test_search_for_workspace_not_intercepted_as_web(self, brain):
        mock_exec = _run(brain, "search for workspace README")
        if mock_exec.called:
            name = mock_exec.call_args[0][0]
            assert name != "search_web", (
                "'search for workspace X' was wrongly routed to search_web"
            )
