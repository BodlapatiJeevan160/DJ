"""
test_phase24_routing_regression.py
===================================
Regression suite for the Phase-24 runtime routing bug.

Bug report
----------
When the user said "Open Notepad", JARVIS responded with:
  "Unfortunately, I'm unable to directly open applications..."

Root cause
----------
"Open Notepad" had no deterministic fast-path in brain.py.  It fell to
LLM Pass-1 (phi3:mini), which sometimes returned {"tool": null}, causing
LLM Pass-2 to generate a confabulated refusal.

Fix
---
Two new fast-path branches (H & I) were added to brain.process_input():
  H: _OPEN_APP_PATTERN  -> open_application
  I: _CLOSE_APP_PATTERN -> close_application

These tests verify:
1.  Pattern matching for every supported phrase / app variant.
2.  Fast-path routing (LLM is never called).
3.  Security boundaries (unknown apps NOT fast-pathed; allowlist respected).
4.  Regression for window-management fast-paths (D/E/F) added in Phase 24.
5.  close_application still requires user confirmation (dangerous flag).

All tests are Ollama-free (LLM fully mocked).
"""

import json
import re
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Module-level pattern helpers (replicate what brain.py compiles)
# ---------------------------------------------------------------------------
from core.brain import (
    _OPEN_APP_PATTERN,
    _CLOSE_APP_PATTERN,
    _APP_ALLOWLIST,
    _WM_LIST_PATTERN,
    _WM_APP_PATTERN,
    _WM_SNAP_PATTERN,
)


# ===========================================================================
# Section 1 - Pattern-level unit tests (no Brain object needed)
# ===========================================================================

class TestOpenAppPattern:
    """_OPEN_APP_PATTERN must match supported verbs and NOT match window-management phrases."""

    @pytest.mark.parametrize("phrase,expected_group", [
        ("Open Notepad",          "notepad"),
        ("open notepad",          "notepad"),
        ("OPEN NOTEPAD",          "notepad"),
        ("Launch Chrome",         "chrome"),
        ("launch chrome",         "chrome"),
        ("Start Calculator",      "calculator"),
        ("start calculator",      "calculator"),
        ("Run Paint",             "paint"),
        ("Open Notepad.",         "notepad"),
        ("  Open Notepad  ",      "notepad"),
        ("Open 7zip",             "7zip"),
        ("open word",             "word"),
        ("open excel",            "excel"),
        ("open discord",          "discord"),
    ])
    def test_matches(self, phrase, expected_group):
        m = _OPEN_APP_PATTERN.match(phrase.strip())
        assert m is not None, f"Pattern should match: {phrase!r}"
        assert m.group(1).strip().lower() == expected_group, (
            f"Expected group(1)={expected_group!r}, got {m.group(1).strip().lower()!r}"
        )

    @pytest.mark.parametrize("phrase", [
        "Minimize Notepad",
        "Search YouTube for Chrome",
        "Open browser",
        "What can you open?",
        "Open",
    ])
    def test_non_app_phrases_not_in_allowlist(self, phrase):
        m = _OPEN_APP_PATTERN.match(phrase.strip())
        if m:
            raw = m.group(1).strip().lower()
            assert raw not in _APP_ALLOWLIST, (
                f"Non-app phrase {phrase!r} matched and {raw!r} is in allowlist - unexpected"
            )


class TestCloseAppPattern:
    """_CLOSE_APP_PATTERN must match close/quit/kill/exit/terminate verbs."""

    @pytest.mark.parametrize("phrase,expected_group", [
        ("Close Notepad",          "notepad"),
        ("close notepad",          "notepad"),
        ("Quit Chrome",            "chrome"),
        ("quit chrome",            "chrome"),
        ("Exit Firefox",           "firefox"),
        ("Kill Spotify",           "spotify"),
        ("Terminate Discord",      "discord"),
        ("Force close Steam",      "steam"),
        ("force close steam",      "steam"),
        ("Close Notepad.",         "notepad"),
        ("  Close Notepad  ",      "notepad"),
    ])
    def test_matches(self, phrase, expected_group):
        m = _CLOSE_APP_PATTERN.match(phrase.strip())
        assert m is not None, f"Pattern should match: {phrase!r}"
        assert m.group(1).strip().lower() == expected_group

    @pytest.mark.parametrize("phrase", [
        "Open Notepad",
        "Minimize Notepad",
        "Close window",
        "Close",
    ])
    def test_non_close_phrases_not_in_allowlist(self, phrase):
        m = _CLOSE_APP_PATTERN.match(phrase.strip())
        if m:
            raw = m.group(1).strip().lower()
            assert raw not in _APP_ALLOWLIST, (
                f"Non-close-app phrase {phrase!r} matched and {raw!r} is in allowlist"
            )


class TestAppAllowlist:
    """_APP_ALLOWLIST must include all expected identifiers and no extras."""

    @pytest.mark.parametrize("identifier", [
        "notepad", "calculator", "calc", "paint", "explorer", "wordpad", "snipping",
        "chrome", "firefox", "edge", "vscode", "code", "blender", "vlc", "spotify",
        "steam", "discord", "word", "excel", "powerpoint", "teams", "zoom", "obs",
        "gimp", "winrar", "7zip",
    ])
    def test_known_apps_in_allowlist(self, identifier):
        assert identifier in _APP_ALLOWLIST, f"{identifier!r} missing from _APP_ALLOWLIST"

    @pytest.mark.parametrize("identifier", [
        "browser", "cmd", "powershell", "regedit", "taskmgr", "malware", "../../",
    ])
    def test_unknown_apps_not_in_allowlist(self, identifier):
        assert identifier not in _APP_ALLOWLIST, (
            f"{identifier!r} must NOT be in _APP_ALLOWLIST"
        )


# ===========================================================================
# Section 2 - Brain fast-path routing tests (LLM fully mocked / never called)
# ===========================================================================

def _ok_result(tool_name, data=None):
    return json.dumps({"success": True, "tool": tool_name, "data": data or {}, "error": None})


class TestOpenApplicationFastPath:
    """
    'Open <app>' must route to open_application without any LLM call.
    This is the primary regression for the bug report.
    """

    @pytest.mark.parametrize("phrase,expected_app", [
        ("Open Notepad",       "notepad"),
        ("open notepad",       "notepad"),
        ("OPEN NOTEPAD",       "notepad"),
        ("Launch Chrome",      "chrome"),
        ("Start Calculator",   "calculator"),
        ("Run Paint",          "paint"),
        ("open notepad.",      "notepad"),
        ("open spotify",       "spotify"),
        ("open discord",       "discord"),
        ("open word",          "word"),
        ("open excel",         "excel"),
        ("open vlc",           "vlc"),
    ])
    def test_open_app_routed_without_llm(self, phrase, expected_app):
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect      = AssertionError("LLM.chat called - fast-path missed")
            MockLLM.return_value.chat_json.side_effect = AssertionError("LLM.chat_json called - fast-path missed")
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(
                brain.registry, "execute_tool",
                return_value=_ok_result("open_application", {"application": expected_app.capitalize()})
            ) as mock_exec:
                brain.process_input(phrase)
                assert mock_exec.called, f"execute_tool was not called for {phrase!r}"
                call_kwargs = mock_exec.call_args[1]
                assert call_kwargs["name"] == "open_application", (
                    f"Expected open_application, got {call_kwargs['name']}"
                )
                assert call_kwargs["args"].get("application") == expected_app, (
                    f"Expected application={expected_app!r}, got {call_kwargs['args']}"
                )

    def test_open_notepad_grounded_response(self):
        """End-to-end: grounded response confirms launch, no confabulation."""
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect      = AssertionError("LLM called")
            MockLLM.return_value.chat_json.side_effect = AssertionError("LLM called")
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(
                brain.registry, "execute_tool",
                return_value=_ok_result("open_application", {"application": "Notepad"})
            ):
                response = brain.process_input("Open Notepad")
        assert "Notepad" in response, f"Response missing 'Notepad': {response!r}"
        assert "unable" not in response.lower(), f"Refusal found in response: {response!r}"
        assert "cannot" not in response.lower(), f"Refusal found in response: {response!r}"
        assert "can't" not in response.lower(),  f"Refusal found in response: {response!r}"

    def test_unknown_app_falls_through_to_llm(self):
        """'Open FakeApp123' must NOT fast-path; falls to LLM Pass-1 instead."""
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat_json.return_value = '{"tool": null, "args": null}'
            MockLLM.return_value.chat.return_value = "That application is not supported."
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(brain.registry, "execute_tool") as mock_exec:
                brain.process_input("Open FakeApp123")
                assert not mock_exec.called, "execute_tool must not be called for unknown app"


class TestCloseApplicationFastPath:
    """'Close/Quit/Kill/Exit/Terminate <app>' must route to close_application."""

    @pytest.mark.parametrize("phrase,expected_app", [
        ("Close Notepad",         "notepad"),
        ("Quit Chrome",           "chrome"),
        ("Exit Firefox",          "firefox"),
        ("Kill Spotify",          "spotify"),
        ("Terminate Discord",     "discord"),
        ("Force close Steam",     "steam"),
    ])
    def test_close_app_routed_without_llm(self, phrase, expected_app):
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect      = AssertionError("LLM called")
            MockLLM.return_value.chat_json.side_effect = AssertionError("LLM called")
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(
                brain.registry, "execute_tool",
                return_value=_ok_result("close_application", {"application": expected_app.capitalize()})
            ) as mock_exec:
                brain.process_input(phrase)
                assert mock_exec.called
                call_kwargs = mock_exec.call_args[1]
                assert call_kwargs["name"] == "close_application"
                assert call_kwargs["args"].get("application") == expected_app

    def test_close_application_confirmation_required(self):
        """close_application is marked dangerous=True - registry must invoke confirm_callback."""
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        deny = lambda name, args: False
        result = registry.execute_tool(
            "close_application", {"application": "notepad"}, confirm_callback=deny
        )
        assert "denied" in result.lower() or "User denied" in result, (
            "close_application must be blocked when user denies"
        )


# ===========================================================================
# Section 3 - Phase 24 window-management fast-path regression
# ===========================================================================

class TestWindowManagementFastPathsUnchanged:
    """Smoke-test the four window-management fast-paths from Phase 24."""

    @pytest.mark.parametrize("phrase,expected_tool,expected_args", [
        ("list windows",              "list_windows",    {}),
        ("show all windows",          "list_windows",    {}),
        ("minimize Notepad",          "minimize_window", {"app_name": "Notepad"}),
        ("maximize Chrome",           "maximize_window", {"app_name": "Chrome"}),
        ("focus VS Code",             "focus_window",    {"app_name": "VS Code"}),
        ("restore Spotify",           "restore_window",  {"app_name": "Spotify"}),
        ("snap Notepad to the left",  "snap_window",     {"app_name": "Notepad", "position": "left"}),
        ("snap Chrome right",         "snap_window",     {"app_name": "Chrome",  "position": "right"}),
    ])
    def test_wm_fast_path(self, phrase, expected_tool, expected_args):
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect      = AssertionError("LLM called")
            MockLLM.return_value.chat_json.side_effect = AssertionError("LLM called")
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(
                brain.registry, "execute_tool",
                return_value=_ok_result(expected_tool, {})
            ) as mock_exec:
                brain.process_input(phrase)
                assert mock_exec.called, f"execute_tool not called for: {phrase!r}"
                call_kwargs = mock_exec.call_args[1]
                assert call_kwargs["name"] == expected_tool, (
                    f"Phrase {phrase!r}: expected {expected_tool}, got {call_kwargs['name']}"
                )
                for key, val in expected_args.items():
                    assert call_kwargs["args"].get(key) == val, (
                        f"Phrase {phrase!r}: expected args[{key!r}]={val!r}, "
                        f"got {call_kwargs['args']}"
                    )


# ===========================================================================
# Section 4 - Collision / disambiguation tests
# ===========================================================================

class TestPatternDisambiguation:
    """
    Phrases that could plausibly be caught by multiple patterns must route
    to the *correct* tool.
    """

    @pytest.mark.parametrize("phrase,expected_tool", [
        # "close Notepad window" has 'window' suffix -> _WM_APP_PATTERN -> close_window
        ("close Notepad window",  "close_window"),
        # "Close Notepad" without 'window' -> _CLOSE_APP_PATTERN -> close_application
        ("Close Notepad",         "close_application"),
        # "Open Notepad" -> _OPEN_APP_PATTERN -> open_application
        ("Open Notepad",          "open_application"),
    ])
    def test_phrase_routes_to_correct_tool(self, phrase, expected_tool):
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect      = AssertionError("LLM called")
            MockLLM.return_value.chat_json.side_effect = AssertionError("LLM called")
            from core.brain import Brain
            brain = Brain()
            brain.history = []
            with patch.object(
                brain.registry, "execute_tool",
                return_value=_ok_result(expected_tool, {})
            ) as mock_exec:
                brain.process_input(phrase)
                assert mock_exec.called, f"execute_tool not called for {phrase!r}"
                actual = mock_exec.call_args[1]["name"]
                assert actual == expected_tool, (
                    f"{phrase!r}: expected {expected_tool}, got {actual}"
                )


# ===========================================================================
# Section 5 - Security boundary tests
# ===========================================================================

class TestSecurityBoundaries:
    """Fast-paths must not weaken existing security protections."""

    def test_open_unknown_app_not_fast_pathed(self):
        """'Open cmd', 'Open regedit', etc. must NOT match the fast-path."""
        for unsafe in ("cmd", "powershell", "regedit", "taskmgr"):
            m = _OPEN_APP_PATTERN.match(f"Open {unsafe}")
            if m:
                assert m.group(1).strip().lower() not in _APP_ALLOWLIST, (
                    f"Unsafe app '{unsafe}' must not be in _APP_ALLOWLIST"
                )

    def test_allowlist_path_traversal_rejected(self):
        """Path-traversal strings must not match the patterns."""
        dangerous = ["../../windows/system32", "notepad.exe", "C:\\Windows\\system32"]
        for d in dangerous:
            m = _OPEN_APP_PATTERN.match(f"Open {d}")
            if m:
                assert m.group(1).strip().lower() not in _APP_ALLOWLIST

    def test_close_window_still_requires_confirmation(self):
        """close_window (WM) confirmation gate is not affected by the new fast-paths."""
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        deny = lambda name, args: False
        result = registry.execute_tool("close_window", {"app_name": "Notepad"}, confirm_callback=deny)
        assert "denied" in result.lower() or "User denied" in result

    def test_open_application_not_dangerous(self):
        """open_application must NOT require user confirmation."""
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["open_application"]
        assert is_dangerous is False, "open_application must not be flagged as dangerous"

    def test_close_application_is_dangerous(self):
        """close_application must require user confirmation."""
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["close_application"]
        assert is_dangerous is True, "close_application must be flagged as dangerous"
