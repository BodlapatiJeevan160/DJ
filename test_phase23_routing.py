"""
test_phase23_routing.py
=======================
Targeted regression tests for Phase 23+ routing fix.

Covers all 11 required cases from the routing fix specification.
"""
import json
import pytest
from unittest.mock import MagicMock
from core.brain import (
    Brain,
    _MEM_RECALL_PATTERN,
    _REMINDER_PATTERN,
    _PCA_WATCH_PATTERN,
    _SCREEN_INTENT_PATTERN,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_brain():
    b = Brain()
    b.history = []
    return b


def _mock_data(name, args):
    return {
        "get_datetime": {"date": "2026-09-23", "time": "19:45", "weekday": "Wednesday"},
        "toggle_speech": {"enabled": args.get("enabled", True)},
        "toggle_voice_input": {"enabled": args.get("enabled", True)},
        "recall": {"key": args.get("key", ""), "value": "Python"},
        "set_reminder": {
            "message": args.get("message", "reminder"),
            "delay_seconds": args.get("delay_seconds", 10),
        },
        "add_context_rule": {
            "keyword": args.get("keyword", ""),
            "message": args.get("message", ""),
        },
        "analyze_screen": {"analysis": "screen result"},
    }.get(name, {})


def _safe_exec(name, args, confirm_callback=None, **kw):
    if confirm_callback:
        confirm_callback(name, args)
    return json.dumps({"success": True, "data": _mock_data(name, args)})


def run_with_tracking(brain, query, force_llm_tool=None, force_llm_response=None):
    """Run query and report whether Ollama (chat_json or chat) was called."""
    ollama_called = {"value": False}
    orig_cj, orig_c = brain.llm.chat_json, brain.llm.chat

    def track_cj(msgs):
        ollama_called["value"] = True
        return json.dumps(force_llm_tool) if force_llm_tool else orig_cj(msgs)

    def track_c(msgs):
        ollama_called["value"] = True
        return force_llm_response if force_llm_response else orig_c(msgs)

    brain.llm.chat_json = track_cj
    brain.llm.chat = track_c
    brain.registry.execute_tool = _safe_exec

    import builtins
    orig_in = builtins.input
    builtins.input = lambda *a, **k: "y"
    try:
        resp = brain.process_input(query)
    finally:
        builtins.input = orig_in
        brain.llm.chat_json = orig_cj
        brain.llm.chat = orig_c
    return resp, ollama_called["value"]


# ===========================================================================
# 1. get_datetime fast-path
# ===========================================================================

class TestGetDatetimeFastPath:
    def test_what_time_is_it(self):
        _, o = run_with_tracking(make_brain(), "What time is it?")
        assert not o, "get_datetime fast-path must NOT call Ollama"

    def test_what_is_the_time(self):
        _, o = run_with_tracking(make_brain(), "What is the time?")
        assert not o, "get_datetime fast-path must NOT call Ollama"

    def test_what_day_is_it(self):
        _, o = run_with_tracking(make_brain(), "What day is it?")
        assert not o, "get_datetime fast-path must NOT call Ollama"


# ===========================================================================
# 2. toggle_speech fast-path
# ===========================================================================

class TestToggleSpeechFastPath:
    def test_turn_speech_on(self):
        _, o = run_with_tracking(make_brain(), "Turn speech on.")
        assert not o, "toggle_speech fast-path must NOT call Ollama"

    def test_turn_voice_off(self):
        _, o = run_with_tracking(make_brain(), "Turn voice off.")
        assert not o, "toggle_speech fast-path must NOT call Ollama"


# ===========================================================================
# 3. Memory / Preference fast-path
# ===========================================================================

MEMORY_QUERIES = [
    "What is my favorite programming language?",
    "What's my favorite programming language?",
    "What's my fav programming language?",
    "What programming language do I like?",
    "Which programming language do I prefer?",
    "What is my preferred editor?",
]


class TestMemoryRecallFastPath:
    @pytest.mark.parametrize("q", MEMORY_QUERIES)
    def test_pattern_matches(self, q):
        assert _MEM_RECALL_PATTERN.match(q.strip()), f"Pattern should match: {q!r}"

    def test_fast_path_no_ollama(self):
        _, o = run_with_tracking(
            make_brain(), "What is my favorite programming language?"
        )
        assert not o, "Memory recall fast-path must NOT call Ollama"

    def test_variant_no_ollama(self):
        for q in [
            "What's my fav programming language?",
            "What programming language do I like?",
            "Which editor do I prefer?",
        ]:
            _, o = run_with_tracking(make_brain(), q)
            assert not o, f"Must not call Ollama for: {q!r}"

    def test_key_normalization(self):
        import re
        adj = re.compile(
            r"^(?:favorite|favourite|fav|preferred|go-to|default|usual)\s+",
            re.IGNORECASE,
        )
        m = _MEM_RECALL_PATTERN.match("What is my favorite programming language?")
        assert m
        raw = (m.group(2) or m.group(3) or m.group(4) or "").strip()
        key = adj.sub("", raw).strip().lower()
        assert key == "programming language", f"Unexpected key: {key!r}"

    def test_recall_routes_through_registry(self):
        brain = make_brain()
        called = {"name": None}

        def tr(name, args, confirm_callback=None, **kw):
            called["name"] = name
            return json.dumps(
                {"success": True, "data": {"key": args.get("key"), "value": "Python"}}
            )

        brain.registry.execute_tool = tr
        brain.process_input("What is my favorite programming language?")
        assert called["name"] == "recall", "Expected recall, got {}".format(called["name"])


# ===========================================================================
# 4. Deterministic reminder fast-path
# ===========================================================================

class TestReminderFastPath:
    @pytest.mark.parametrize(
        "q,amt,unit",
        [
            ("Remind me in 10 seconds.", 10, "second"),
            ("Remind me in 5 minutes.", 5, "minute"),
            ("Remind me in 2 hours.", 2, "hour"),
            ("Remind me in 1 second to breathe.", 1, "second"),
            ("Remind me in 30 minutes to take a break.", 30, "minute"),
        ],
    )
    def test_pattern_matches(self, q, amt, unit):
        m = _REMINDER_PATTERN.match(q.strip())
        assert m and int(m.group(1)) == amt and m.group(2).lower() == unit

    @pytest.mark.parametrize(
        "q",
        [
            "Remind me after lunch.",
            "Remind me later.",
            "Remind me at 5pm.",
            "Can you remind me in ten seconds?",
        ],
    )
    def test_ambiguous_not_matched(self, q):
        assert _REMINDER_PATTERN.match(q.strip()) is None

    def test_10_seconds_no_ollama(self):
        _, o = run_with_tracking(make_brain(), "Remind me in 10 seconds.")
        assert not o

    def test_5_minutes_no_ollama(self):
        _, o = run_with_tracking(make_brain(), "Remind me in 5 minutes.")
        assert not o

    def test_delay_conversion_3_minutes(self):
        brain = make_brain()
        recv = {}

        def tr(name, args, confirm_callback=None, **kw):
            if name == "set_reminder":
                recv.update(args)
            return json.dumps({"success": True, "data": args})

        brain.registry.execute_tool = tr
        brain.process_input("Remind me in 3 minutes.")
        assert recv.get("delay_seconds") == 180

    def test_ambiguous_reminder_is_llm_bound(self):
        _, o = run_with_tracking(
            make_brain(),
            "Remind me after lunch.",
            force_llm_tool={
                "tool": "set_reminder",
                "args": {"message": "lunch", "delay_seconds": 3600},
            },
            force_llm_response="Reminder set.",
        )
        assert o, "Ambiguous reminder must call Ollama"


# ===========================================================================
# 5. PCA context-rule fast-path
# ===========================================================================

PCA_QUERIES = [
    "Watch for Notepad and notify me.",
    "Monitor Notepad and notify me.",
    "Notify me when I'm working in Chrome.",
    "Alert me when Notepad is open.",
    "Watch for Chrome and notify me when I'm working in it.",
    "Watch for Notepad and notify me when I'm working in it.",
    "Watch for Chrome.",
    "When I'm working in Chrome, notify me.",
]

NON_PCA_QUERIES = [
    "Why is the sky blue?",
    "Open Notepad.",
    "Close Chrome.",
    "What time is it?",
]


class TestPCAContextRuleFastPath:
    @pytest.mark.parametrize("q", PCA_QUERIES)
    def test_pattern_matches(self, q):
        assert _PCA_WATCH_PATTERN.match(q.strip()), f"PCA pattern should match: {q!r}"

    @pytest.mark.parametrize("q", NON_PCA_QUERIES)
    def test_non_pca_not_matched(self, q):
        assert _PCA_WATCH_PATTERN.match(q.strip()) is None, f"Should NOT match: {q!r}"

    def test_notepad_no_ollama(self):
        _, o = run_with_tracking(make_brain(), "Watch for Notepad and notify me.")
        assert not o

    def test_chrome_no_ollama(self):
        _, o = run_with_tracking(
            make_brain(), "Notify me when I'm working in Chrome."
        )
        assert not o

    def test_routes_through_registry(self):
        brain = make_brain()
        called = {"name": None, "args": {}}

        def tr(name, args, confirm_callback=None, **kw):
            called["name"] = name
            called["args"] = args
            return json.dumps(
                {
                    "success": True,
                    "data": {
                        "keyword": args.get("keyword", ""),
                        "message": args.get("message", ""),
                    },
                }
            )

        brain.registry.execute_tool = tr
        brain.process_input("Watch for Notepad and notify me.")
        assert called["name"] == "add_context_rule"
        assert "notepad" in called["args"].get("keyword", "").lower()

    def test_monitor_variant_no_ollama(self):
        _, o = run_with_tracking(make_brain(), "Monitor Notepad and notify me.")
        assert not o

    def test_chrome_keyword_no_trailing_punctuation(self):
        """Keyword extracted from \"Notify me when I'm working in Chrome.\" must be 'Chrome', not 'Chrome.'"""
        brain = make_brain()
        captured = {"keyword": None}

        def tr(name, args, confirm_callback=None, **kw):
            captured["keyword"] = args.get("keyword", "")
            return json.dumps({
                "success": True,
                "data": {"keyword": args.get("keyword", ""), "message": args.get("message", "")},
            })

        brain.registry.execute_tool = tr
        brain.process_input("Notify me when I'm working in Chrome.")
        assert captured["keyword"] == "Chrome", (
            f"Keyword should be 'Chrome' without trailing punctuation, got {captured['keyword']!r}"
        )

    def test_natural_pca_extraction_strips_suffix(self):
        """Keyword extracted from \"Watch for Chrome and notify me when I'm working in it.\" must be 'Chrome'"""
        brain = make_brain()
        captured = {"keyword": None}

        def tr(name, args, confirm_callback=None, **kw):
            captured["keyword"] = args.get("keyword", "")
            return json.dumps({
                "success": True,
                "data": {"keyword": args.get("keyword", ""), "message": args.get("message", "")},
            })

        brain.registry.execute_tool = tr
        brain.process_input("Watch for Chrome and notify me when I'm working in it.")
        assert captured["keyword"].lower() == "chrome", (
            f"Keyword should be 'Chrome', got {captured['keyword']!r}"
        )


# ===========================================================================
# 6 & 7. analyze_screen guard -- NEVER fires for general knowledge
# ===========================================================================

SCREEN_LEGIT = [
    "Analyze my screen.",
    "What is on my screen?",
    "Read the text on my screen.",
    "Describe what you see on my display.",
    "What's visible right now?",
]

SCREEN_GENERAL = [
    "Why is the sky blue?",
    "What is Python?",
    "Explain quantum physics.",
    "How does photosynthesis work?",
    "Tell me about databases.",
]


class TestAnalyzeScreenGuard:
    @pytest.mark.parametrize("q", SCREEN_LEGIT)
    def test_intent_pattern_matches_legit(self, q):
        assert _SCREEN_INTENT_PATTERN.search(q), f"Should match: {q!r}"

    @pytest.mark.parametrize("q", SCREEN_GENERAL)
    def test_intent_pattern_no_match_general(self, q):
        assert not _SCREEN_INTENT_PATTERN.search(q), f"Should NOT match: {q!r}"

    def _guard_test(self, query):
        brain = make_brain()
        brain.llm.chat_json = MagicMock(
            return_value=json.dumps(
                {"tool": "analyze_screen", "args": {"query": query}}
            )
        )
        brain.llm.chat = MagicMock(return_value="General answer.")
        tools = []

        def tr(name, args, confirm_callback=None, **kw):
            tools.append(name)
            return json.dumps({"success": True, "data": {}})

        brain.registry.execute_tool = tr
        brain.process_input(query)
        return tools

    def test_sky_blue_no_analyze_screen(self):
        assert "analyze_screen" not in self._guard_test("Why is the sky blue?")

    def test_what_is_python_no_analyze_screen(self):
        assert "analyze_screen" not in self._guard_test("What is Python?")

    def test_quantum_no_analyze_screen(self):
        assert "analyze_screen" not in self._guard_test("Explain quantum physics.")

    def test_photosynthesis_no_analyze_screen(self):
        assert "analyze_screen" not in self._guard_test("How does photosynthesis work?")


# ===========================================================================
# 8 & 9. analyze_screen IS available for explicit screen requests
# ===========================================================================

class TestAnalyzeScreenAllowed:
    def _allowed_test(self, query):
        brain = make_brain()
        brain.llm.chat_json = MagicMock(
            return_value=json.dumps(
                {"tool": "analyze_screen", "args": {"query": query}}
            )
        )
        brain.llm.chat = MagicMock(return_value="Screen analyzed.")
        tools = []

        def tr(name, args, confirm_callback=None, **kw):
            if confirm_callback:
                confirm_callback(name, args)
            tools.append(name)
            return json.dumps({"success": True, "data": {"analysis": "ok"}})

        brain.registry.execute_tool = tr

        import builtins
        orig = builtins.input
        builtins.input = lambda *a, **k: "y"
        try:
            brain.process_input(query)
        finally:
            builtins.input = orig
        return tools

    def test_analyze_my_screen(self):
        assert "analyze_screen" in self._allowed_test("Analyze my screen.")

    def test_read_my_screen(self):
        assert "analyze_screen" in self._allowed_test("Read the text on my screen.")

    def test_what_is_on_screen(self):
        assert "analyze_screen" in self._allowed_test("What is on my screen?")


# ===========================================================================
# 10. Dangerous tools still require confirmation
# ===========================================================================

class TestDangerousToolConfirmation:
    def test_write_file_requires_confirmation(self):
        brain = make_brain()
        brain.llm.chat_json = MagicMock(
            return_value=json.dumps(
                {
                    "tool": "write_file",
                    "args": {
                        "location": "desktop",
                        "name": "test.txt",
                        "content": "hello",
                    },
                }
            )
        )
        brain.llm.chat = MagicMock(return_value="Denied.")
        cc = {"value": False}

        def tr(name, args, confirm_callback=None, **kw):
            if name == "write_file" and confirm_callback:
                cc["value"] = True
                confirm_callback(name, args)
            return json.dumps({"success": False, "error": "User denied."})

        brain.registry.execute_tool = tr
        
        import builtins
        orig_in = builtins.input
        builtins.input = lambda *a, **k: "y"
        try:
            brain.process_input("Write hello to desktop test.txt")
        finally:
            builtins.input = orig_in
            
        assert cc["value"], "write_file must invoke confirm_callback"

    def test_fast_path_tools_not_dangerous(self):
        from core.tool_registry import ToolRegistry

        reg = ToolRegistry()
        for tool in [
            "get_datetime",
            "toggle_speech",
            "recall",
            "set_reminder",
            "add_context_rule",
        ]:
            if tool in reg.tools:
                _, is_dangerous = reg.tools[tool]
                assert not is_dangerous, f"Fast-pathed tool {tool!r} must NOT be dangerous"


# ===========================================================================
# 11. Ambiguous requests remain LLM-bound
# ===========================================================================

class TestAmbiguousLLMBound:
    def test_ambiguous_time_uses_llm(self):
        brain = make_brain()
        called = {"value": False}
        orig_cj = brain.llm.chat_json

        def track(msgs):
            called["value"] = True
            return '{"tool": "get_datetime", "args": {}}'

        brain.llm.chat_json = track
        brain.llm.chat = MagicMock(return_value="5pm Tokyo.")
        brain.registry.execute_tool = _safe_exec
        
        import builtins
        orig_in = builtins.input
        builtins.input = lambda *a, **k: "y"
        try:
            brain.process_input("What time is it in Tokyo?")
        finally:
            builtins.input = orig_in
            
        assert called["value"], "Ambiguous time query must use LLM"

    def test_ambiguous_reminder_uses_llm(self):
        _, o = run_with_tracking(
            make_brain(),
            "Remind me about the meeting.",
            force_llm_tool={
                "tool": "set_reminder",
                "args": {"message": "meeting", "delay_seconds": 3600},
            },
            force_llm_response="Reminder set.",
        )
        assert o, "Ambiguous reminder must call Ollama"
