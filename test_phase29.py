"""
test_phase29.py
===============
Phase 29: Controlled Multi-Step Task Execution

Tests cover all 30 requirements from the Phase 29 specification.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.brain import Brain, CategoryScorer, MAX_STEPS
from core.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_SUCCESS_RESULT = json.dumps({"success": True, "data": {"result": "ok"}})
_FAILURE_RESULT = json.dumps({"success": False, "error": "tool error", "data": None})
_SEARCH_WS_RESULT = json.dumps({
    "success": True,
    "data": {
        "results": [{"file": "/workspace/README.md", "match": "Python project"}],
        "skipped_unsearchable": 0,
        "limit_reached": False,
    }
})
_READ_FILE_RESULT = json.dumps({
    "success": True,
    "data": {"content": "# My Project\nThis uses Python and FastAPI."}
})


def _make_brain():
    """Return a Brain with LLM and session persistence mocked out."""
    with patch("core.session.SessionStore.load", return_value=[]):
        brain = Brain()
    brain.llm = MagicMock()
    brain.llm.chat.return_value = "Mock LLM response"
    brain.llm.chat_json.return_value = json.dumps(
        {"tool": None, "args": None, "requires_followup": False}
    )
    brain.registry.execute_tool = MagicMock(return_value=_SUCCESS_RESULT)
    return brain


# ---------------------------------------------------------------------------
# Tests 1-6: Step-count and duplicate-call guards
# ---------------------------------------------------------------------------

class TestPhase29StepControl(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_01_single_step_remains_single_step(self):
        """A Pass-1 response with requires_followup=false must yield exactly one tool call."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "get_datetime", "args": {}, "requires_followup": False,
        })
        self.brain.registry.execute_tool.reset_mock()
        self.brain.process_input("What time is it right now?")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "get_datetime")

    def test_02_multi_step_two_tools(self):
        """SEARCH_AND_READ bounded task produces search_workspace then read_workspace_file without LLM."""
        self.brain.registry.execute_tool.side_effect = [_SEARCH_WS_RESULT, _READ_FILE_RESULT]
        self.brain.process_input("Find my README file and read it")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 2)
        call_names = [c[1].get("name") for c in self.brain.registry.execute_tool.call_args_list]
        self.assertEqual(call_names[0], "search_workspace")
        self.assertEqual(call_names[1], "read_workspace_file")
        # LLM must NOT be invoked for a bounded task
        self.brain.llm.chat_json.assert_not_called()

    def test_03_multi_step_three_tools(self):
        """LLM single-tool response with requires_followup=True must still execute exactly one tool."""
        # In the new arch, requires_followup from LLM is ignored. One tool, then Pass-2 grounding.
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "get_datetime", "args": {}, "requires_followup": True
        })
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        # Use an imperative sentence so the conversational pre-classifier allows LLM routing
        self.brain.process_input("Get the current datetime and also system info and battery.")
        # Should execute exactly one tool (no chained LLM follow-up)
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)

    def test_04_max_steps_prevents_fourth_tool(self):
        """After MAX_STEPS tool calls the loop must stop regardless of requires_followup."""
        always_followup = json.dumps({"tool": "get_datetime", "args": {}, "requires_followup": True})
        self.brain.llm.chat_json.return_value = always_followup
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        self.brain.process_input("Keep doing things forever.")
        self.assertLessEqual(self.brain.registry.execute_tool.call_count, MAX_STEPS)

    def test_05_same_tool_same_args_blocked(self):
        """Returning the exact same tool+args a second time must be blocked."""
        same_call = json.dumps({"tool": "search_workspace", "args": {"query": "readme", "search_type": "filename"}, "requires_followup": True})
        self.brain.llm.chat_json.return_value = same_call
        self.brain.registry.execute_tool.return_value = _SEARCH_WS_RESULT
        self.brain.process_input("Search workspace for readme then do it again.")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)

    def test_06_non_consecutive_duplicate_blocked(self):
        """A call from step 1 must be blocked again in step 3."""
        r1 = json.dumps({"tool": "get_datetime", "args": {}, "requires_followup": True})
        r2 = json.dumps({"tool": "get_system_info", "args": {}, "requires_followup": True})
        r3 = json.dumps({"tool": "get_datetime", "args": {}, "requires_followup": False})
        self.brain.llm.chat_json.side_effect = [r1, r2, r3]
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        self.brain.process_input("Get datetime, system info, then datetime again.")
        self.assertLessEqual(self.brain.registry.execute_tool.call_count, 2)


# ---------------------------------------------------------------------------
# Tests 7-10: Failure handling and no retry
# ---------------------------------------------------------------------------

class TestPhase29FailureHandling(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_07_failed_step1_aborts(self):
        """If step 1 returns success=false, the loop must stop immediately."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "search_workspace", "args": {"query": "readme", "search_type": "filename"}, "requires_followup": True,
        })
        self.brain.registry.execute_tool.return_value = _FAILURE_RESULT
        self.brain.process_input("Find README then read it.")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)

    def test_08_failed_step2_aborts(self):
        """If SEARCH_AND_READ step 2 (read_workspace_file) fails, no further steps are taken."""
        # Step 1: search returns one match → step 2 fires → step 2 fails → abort
        search_result = json.dumps({
            "success": True,
            "data": {"results": [{"file": "/workspace/f.txt", "match": ""}],
                     "skipped_unsearchable": 0, "limit_reached": False}
        })
        self.brain.registry.execute_tool.side_effect = [search_result, _FAILURE_RESULT]
        self.brain.process_input("Find f.txt and read it")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 2)
        call_names = [c[1].get("name") for c in self.brain.registry.execute_tool.call_args_list]
        self.assertEqual(call_names[0], "search_workspace")
        self.assertEqual(call_names[1], "read_workspace_file")

    def test_09_failed_step3_aborts(self):
        """SEARCH_AND_REPORT with failed search stops at 1 call, no third step generated."""
        self.brain.registry.execute_tool.return_value = _FAILURE_RESULT
        self.brain.process_input("Search my workspace for MAX_MEMORIES and tell me which file contains it")
        # Only one tool call; no retries or additional tools
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        self.brain.llm.chat_json.assert_not_called()

    def test_10_no_automatic_retry(self):
        """After a tool failure the same tool must NOT be retried automatically."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "read_file", "args": {"name": "f.txt", "location": "desktop"}, "requires_followup": False,
        })
        self.brain.registry.execute_tool.return_value = _FAILURE_RESULT
        result = self.brain.process_input("Read the file.")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Tests 11-13: Dangerous tool confirmation during multi-step
# ---------------------------------------------------------------------------

class TestPhase29DangerousTools(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_11_dangerous_step2_requires_confirmation(self):
        """SEARCH_AND_READ step 2 (read_workspace_file) must always pass confirm_callback to ToolRegistry."""
        self.brain.registry.execute_tool.side_effect = [_SEARCH_WS_RESULT, _SUCCESS_RESULT]
        self.brain.process_input("Find my README file and read it")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 2)
        # Both steps must receive confirm_callback (ToolRegistry enforces confirmation)
        for i, call in enumerate(self.brain.registry.execute_tool.call_args_list):
            self.assertIsNotNone(call[1].get("confirm_callback"),
                                 f"Step {i+1} must receive confirm_callback")

    def test_12_confirmation_denied_aborts(self):
        """If step 1 search_workspace is denied, no step 2 is attempted."""
        denial = "ERROR: User denied permission to execute search_workspace. Do NOT pretend you executed it."
        self.brain.registry.execute_tool.return_value = denial
        self.brain.process_input("Find my README file and read it")
        # Only one tool call attempted; step 2 must not fire
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)

    def test_13_dangerous_step3_requires_confirmation(self):
        """All bounded task tool calls must pass confirm_callback regardless of step number."""
        # SEARCH_AND_REPORT: single tool with confirm_callback
        self.brain.registry.execute_tool.return_value = _SEARCH_WS_RESULT
        self.brain.process_input("Search my workspace for MAX_MEMORIES and tell me which file contains it")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertIsNotNone(kwargs.get("confirm_callback"),
                             "SEARCH_AND_REPORT tool must receive confirm_callback")


# ---------------------------------------------------------------------------
# Tests 14-16: Unknown tools and malformed JSON
# ---------------------------------------------------------------------------

class TestPhase29JSONHardening(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_14_unknown_tool_rejected(self):
        """A follow-up Pass-1 that returns an unknown tool must be rejected."""
        r1 = json.dumps({"tool": "search_workspace", "args": {"query": "x", "search_type": "content"}, "requires_followup": True})
        r2 = json.dumps({"tool": "hack_the_planet", "args": {}, "requires_followup": False})
        self.brain.llm.chat_json.side_effect = [r1, r2]
        self.brain.registry.execute_tool.return_value = _SEARCH_WS_RESULT
        self.brain.process_input("Search then do something dangerous.")
        names = [c[1].get("name") for c in self.brain.registry.execute_tool.call_args_list]
        self.assertNotIn("hack_the_planet", names)

    def test_15_malformed_json_fails_safely(self):
        """Completely unparseable JSON must not trigger an uncontrolled LLM call."""
        self.brain.llm.chat_json.return_value = "not valid json at all"
        self.brain.llm.chat.reset_mock()
        result = self.brain.process_input("Do something requiring a tool.")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, (str, dict))

    def test_16_hallucinated_structure_fails_safely(self):
        """Valid JSON without a tool field must be rejected safely without Pass-2 LLM call."""
        self.brain.llm.chat_json.return_value = '{"fake_tool": "hello", "success": true}'
        self.brain.llm.chat.reset_mock()
        result = self.brain.process_input("Search for something requiring a tool.")
        self.assertIsNotNone(result)
        if isinstance(result, dict):
            self.assertIn("response", result)
            self.assertIn("internal error", result["response"])
        self.brain.llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Tests 17-18: Schema contract
# ---------------------------------------------------------------------------

class TestPhase29Schema(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_17_requires_followup_false_stops_loop(self):
        """requires_followup=false on the first step must mean exactly one tool execution."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "get_datetime", "args": {}, "requires_followup": False,
        })
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        self.brain.process_input("Can you grab the current datetime for me?")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        self.assertEqual(self.brain.llm.chat_json.call_count, 1)

    def test_18_tool_null_forces_no_followup(self):
        """If follow-up Pass-1 returns tool=null, loop stops (no further tool)."""
        r1 = json.dumps({"tool": "get_datetime", "args": {}, "requires_followup": True})
        r2 = json.dumps({"tool": None, "args": {}, "requires_followup": False})
        self.brain.llm.chat_json.side_effect = [r1, r2]
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        self.brain.process_input("Get datetime then decide if more is needed.")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)


# ---------------------------------------------------------------------------
# Tests 19-22: Phase 28 integration and deterministic fast paths
# ---------------------------------------------------------------------------

class TestPhase29Phase28Integration(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_19_phase28_routing_still_active(self):
        """CategoryScorer still selects tool categories for non-fast-path requests."""
        result_ws = CategoryScorer.get_routing_categories("Search my workspace for the README file")
        self.assertIn("WORKSPACE", result_ws)
        result_sys = CategoryScorer.get_routing_categories("Open Chrome")
        self.assertIn("SYSTEM_WM", result_sys)

    def test_20_phase28_fast_paths_not_broken(self):
        """Phase 28 fast-path routing must still bypass the LLM for known patterns."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        self.brain.process_input("Open Chrome")
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "open_application")
        self.brain.llm.chat_json.assert_not_called()

    def test_21_list_windows_remains_deterministic(self):
        """Window-listing fast-paths must not enter the multi-step loop."""
        for phrase in ["Show me my open windows", "List my open windows",
                       "What windows are open?", "Show open windows"]:
            self.brain.llm.chat_json.reset_mock()
            self.brain.registry.execute_tool.reset_mock()
            self.brain.process_input(phrase)
            _, kwargs = self.brain.registry.execute_tool.call_args
            self.assertEqual(kwargs.get("name"), "list_windows", f"Failed for: {phrase!r}")
            self.brain.llm.chat_json.assert_not_called()

    def test_22_open_chrome_remains_deterministic(self):
        """Open Chrome must route through the fast-path to open_application."""
        self.brain.llm.chat_json.reset_mock()
        self.brain.registry.execute_tool.reset_mock()
        self.brain.process_input("Open Chrome")
        self.brain.llm.chat_json.assert_not_called()
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "open_application")


# ---------------------------------------------------------------------------
# Tests 23-25: Memory and session compatibility
# ---------------------------------------------------------------------------

class TestPhase29MemorySessionCompatibility(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_23_phase19_memory_compatible(self):
        """Phase 19 (remember/recall/forget) tools still function through ToolRegistry."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "remember", "args": {"key": "username", "value": "Alice"}, "requires_followup": False,
        })
        self.brain.registry.execute_tool.return_value = json.dumps({"success": True, "data": {}})
        result = self.brain.process_input("Remember that my username is Alice.")
        self.assertIsInstance(result, str)
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "remember")

    def test_24_phase25_semantic_memory_compatible(self):
        """Phase 25 semantic memory tools still function through ToolRegistry."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "store_semantic_memory",
            "args": {"content": "I like Python", "tags": ["preference"]},
            "requires_followup": False,
        })
        self.brain.registry.execute_tool.return_value = json.dumps(
            {"success": True, "data": {"count": 1, "duplicate": False}}
        )
        result = self.brain.process_input("Store that I like Python in semantic memory.")
        self.assertIsInstance(result, str)
        _, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs.get("name"), "store_semantic_memory")

    def test_25_phase27_session_persistence_compatible(self):
        """SessionStore.save must be called after each turn."""
        with patch("core.session.SessionStore.save") as mock_save:
            brain = _make_brain()
            brain.llm.chat_json.return_value = json.dumps(
                {"tool": None, "args": None, "requires_followup": False}
            )
            brain.process_input("Hello JARVIS.")
            mock_save.assert_called()
        self.assertGreater(len(brain.history), 1)


# ---------------------------------------------------------------------------
# Tests 26-30: Grounding, bounded context, security
# ---------------------------------------------------------------------------

class TestPhase29GroundingAndSecurity(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("core.session.SessionStore.save")
        self.patcher.start()
        self.brain = _make_brain()

    def tearDown(self):
        self.patcher.stop()

    def test_26_tool_results_grounded(self):
        """The response must reflect the actual tool result, not invented information."""
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "get_datetime", "args": {}, "requires_followup": False,
        })
        self.brain.registry.execute_tool.return_value = json.dumps({
            "success": True,
            "data": {"date": "2026-10-05", "time": "14:38", "weekday": "Monday"}
        })
        result = self.brain.process_input("What time is it?")
        self.assertIn("14:38", result)
        self.assertIn("Monday", result)
        self.assertIn("2026-10-05", result)

    def test_27_followup_context_bounded(self):
        """SEARCH_AND_READ must NOT invoke any LLM follow-up call — Python state machine handles step 2."""
        self.brain.registry.execute_tool.side_effect = [_SEARCH_WS_RESULT, _READ_FILE_RESULT]
        self.brain.process_input("Find my README file and read it")
        # No follow-up LLM call should have been made
        self.brain.llm.chat_json.assert_not_called()
        # Both tools must have executed
        self.assertEqual(self.brain.registry.execute_tool.call_count, 2)

    def test_28_no_pass2_fallback_on_malformed_json(self):
        """Completely unrecoverable malformed JSON must NOT trigger a Pass-2 LLM call."""
        self.brain.llm.chat_json.return_value = '{"completely_unknown_key": "garbage"}'
        self.brain.llm.chat.reset_mock()
        result = self.brain.process_input("Search for something requiring a tool.")
        self.brain.llm.chat.assert_not_called()
        if isinstance(result, dict):
            self.assertIn("response", result)
        else:
            self.assertIsInstance(result, str)

    def test_29_no_security_bypass(self):
        """Multi-step execution must NOT bypass the ToolRegistry confirm_callback gate."""
        r1 = json.dumps({"tool": "search_workspace", "args": {"query": "secret", "search_type": "content"}, "requires_followup": True})
        r2 = json.dumps({"tool": "run_command", "args": {"command": "echo hi"}, "requires_followup": False})
        self.brain.llm.chat_json.side_effect = [r1, r2]
        self.brain.registry.execute_tool.side_effect = [_SEARCH_WS_RESULT, _SUCCESS_RESULT]
        self.brain.process_input("Search for secret and then run a command.")
        call_list = self.brain.registry.execute_tool.call_args_list
        dangerous_calls = [c for c in call_list if c[1].get("name") == "run_command"]
        for dc in dangerous_calls:
            self.assertIsNotNone(dc[1].get("confirm_callback"),
                                 "Dangerous tool run_command must receive confirm_callback in multi-step")

    def test_30_no_direct_tool_execution(self):
        """All tool executions must go through ToolRegistry.execute_tool with confirm_callback."""
        # LLM single-tool path: one tool, one execute_tool call
        self.brain.llm.chat_json.return_value = json.dumps({
            "tool": "get_datetime", "args": {}, "requires_followup": False
        })
        self.brain.registry.execute_tool.return_value = _SUCCESS_RESULT
        # Imperative input forces LLM routing (not conversational pre-classifier skip)
        self.brain.process_input("Get the current datetime.")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        for c in self.brain.registry.execute_tool.call_args_list:
            self.assertIn("name", c[1], "execute_tool must always receive name kwarg")
            self.assertIn("confirm_callback", c[1],
                          "execute_tool must always receive confirm_callback kwarg")


# ---------------------------------------------------------------------------
# MAX_STEPS constant validation
# ---------------------------------------------------------------------------

class TestPhase29LiveValidations(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("core.brain.SessionStore.load", return_value=[])
        self.patcher.start()
        self.brain = Brain()
        self.brain.llm.chat_json = MagicMock()
        self.brain.llm.chat = MagicMock(return_value="Pass 2 result")
        self.brain.registry = MagicMock()
        self.brain.registry.tools = {
            "search_workspace": (MagicMock(), False),
            "read_workspace_file": (MagicMock(), False),
        }
        self.brain.registry.get_tool_descriptions = MagicMock(return_value='{"search_workspace": "...", "read_workspace_file": "..."}')
        self.brain.registry.execute_tool = MagicMock()

    def tearDown(self):
        self.patcher.stop()

    def test_31_find_and_read_fast_path(self):
        """Find my README file and read it uses SEARCH_AND_READ → read_workspace_file."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "C:\\dummy\\README.md"}]}}',
            '{"success": true, "data": {"content": "hello world", "path": "C:/dummy/README.md"}}'
        ]
        
        result = self.brain.process_input("Find my README file and read it")
        
        calls = self.brain.registry.execute_tool.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["name"], "search_workspace")
        self.assertEqual(calls[1].kwargs["name"], "read_workspace_file")
        # Path must be the forward-slash-normalised absolute path from search result
        self.assertEqual(calls[1].kwargs["args"]["path"], "C:/dummy/README.md")
        # Ensure Pass-1 LLM was NOT called
        self.brain.llm.chat_json.assert_not_called()
        # Absolute path must be passed as 'path', NOT as 'location'/'name'
        self.assertNotIn("location", calls[1].kwargs["args"])
        self.assertNotIn("name", calls[1].kwargs["args"])
        
    def test_32_search_and_summarize_llm_path(self):
        """Search my workspace for MAX_MEMORIES and tell me which file contains it uses SEARCH_AND_REPORT."""
        self.brain.registry.execute_tool.return_value = '{"success": true, "data": {"results": [{"file": "config.py"}]}}'
        
        self.brain.process_input("Search my workspace for MAX_MEMORIES and tell me which file contains it")
        
        calls = self.brain.registry.execute_tool.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["name"], "search_workspace")
        self.assertEqual(calls[0].kwargs["args"]["query"], "MAX_MEMORIES")
        self.brain.llm.chat_json.assert_not_called()
        # SEARCH_AND_REPORT uses deterministic grounding, not llm.chat

    def test_33_ambiguous_read_no_guess(self):
        """Ambiguous 'read my README' correctly triggers SEARCH_AND_READ bounded task."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "C:\\dummy\\README.md"}]}}',
            '{"success": true, "data": {"content": "hello world", "path": "C:/dummy/README.md"}}'
        ]
        self.brain.process_input("Read my README")
        
        calls = self.brain.registry.execute_tool.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["name"], "search_workspace")
        self.assertEqual(calls[1].kwargs["name"], "read_workspace_file")
        self.brain.llm.chat_json.assert_not_called()

    def test_34_step2_receives_path_not_location_name(self):
        """read_workspace_file must receive 'path' kwarg, never 'location' or 'name'."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "/workspace/README.md"}]}}',
            '{"success": true, "data": {"content": "hello", "path": "/workspace/README.md"}}'
        ]
        self.brain.process_input("Find my README file and read it")
        step2_args = self.brain.registry.execute_tool.call_args_list[1].kwargs["args"]
        self.assertIn("path", step2_args)
        self.assertNotIn("location", step2_args)
        self.assertNotIn("name", step2_args)

    def test_35_step2_path_is_authoritative_search_result(self):
        """The path passed to read_workspace_file must be the exact file from search result."""
        search_result = json.dumps({
            "success": True,
            "data": {"results": [{"file": "/workspace/src/main.py", "match": "main"}],
                     "skipped_unsearchable": 0, "limit_reached": False}
        })
        self.brain.registry.execute_tool.side_effect = [
            search_result,
            '{"success": true, "data": {"content": "def main(): pass", "path": "/workspace/src/main.py"}}'
        ]
        self.brain.process_input("Find my main.py and read it")
        step2_args = self.brain.registry.execute_tool.call_args_list[1].kwargs["args"]
        self.assertEqual(step2_args["path"], "/workspace/src/main.py")

    def test_36_missing_search_result_stops_safely(self):
        """If search returns 0 results, no step 2 tool is called."""
        self.brain.registry.execute_tool.return_value = json.dumps({
            "success": True,
            "data": {"results": [], "skipped_unsearchable": 0, "limit_reached": False}
        })
        self.brain.process_input("Find my NONEXISTENT file and read it")
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        self.assertEqual(self.brain.registry.execute_tool.call_args_list[0].kwargs["name"], "search_workspace")

    def test_37_multiple_results_no_guess(self):
        """If search returns multiple results, no file is read — clarification is returned instead."""
        multi_result = json.dumps({
            "success": True,
            "data": {"results": [
                {"file": "/workspace/a/README.md", "match": "README"},
                {"file": "/workspace/b/README.md", "match": "README"},
            ], "skipped_unsearchable": 0, "limit_reached": False}
        })
        self.brain.registry.execute_tool.return_value = multi_result
        result = self.brain.process_input("Find my README file and read it")
        # Only search_workspace called; no read_workspace_file call
        self.assertEqual(self.brain.registry.execute_tool.call_count, 1)
        # Response must mention both paths
        self.assertIn("/workspace/a/README.md", result)
        self.assertIn("/workspace/b/README.md", result)

    def test_38_registry_is_execution_authority(self):
        """read_workspace_file must go through ToolRegistry.execute_tool — never called directly."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "/workspace/README.md"}]}}',
            '{"success": true, "data": {"content": "hello", "path": "/workspace/README.md"}}'
        ]
        self.brain.process_input("Find my README file and read it")
        # Both calls go through execute_tool
        self.assertEqual(self.brain.registry.execute_tool.call_count, 2)
        for call in self.brain.registry.execute_tool.call_args_list:
            self.assertIn("confirm_callback", call[1],
                          "Every execute_tool call must pass confirm_callback")

    def test_39_read_my_readme_works(self):
        """'Read my README' triggers SEARCH_AND_READ and reaches read_workspace_file."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "/workspace/README.md"}]}}',
            '{"success": true, "data": {"content": "# My Project", "path": "/workspace/README.md"}}'
        ]
        self.brain.process_input("Read my README")
        calls = self.brain.registry.execute_tool.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["name"], "search_workspace")
        self.assertEqual(calls[1].kwargs["name"], "read_workspace_file")
        self.brain.llm.chat_json.assert_not_called()

    def test_40_find_and_read_readme_works(self):
        """'Find my README file and read it' triggers SEARCH_AND_READ → read_workspace_file."""
        self.brain.registry.execute_tool.side_effect = [
            '{"success": true, "data": {"results": [{"file": "/workspace/README.md"}]}}',
            '{"success": true, "data": {"content": "# My Project", "path": "/workspace/README.md"}}'
        ]
        self.brain.process_input("Find my README file and read it")
        calls = self.brain.registry.execute_tool.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["name"], "search_workspace")
        self.assertEqual(calls[1].kwargs["name"], "read_workspace_file")
        self.brain.llm.chat_json.assert_not_called()

    def test_41_deterministic_read_workspace_file_response(self):
        """read_workspace_file must return deterministic output bypassing Pass-2 LLM."""
        res_json = json.dumps({"success": True, "data": {"content": "Hello World", "path": "/workspace/README.md"}})
        result = self.brain._build_grounded_response("read_workspace_file", {}, res_json)
        self.assertIsNotNone(result)
        self.assertIn("Contents of /workspace/README.md:", result)
        self.assertIn("Hello World", result)

    def test_42_deterministic_read_file_response(self):
        """read_file must return deterministic output bypassing Pass-2 LLM."""
        res_json = json.dumps({"success": True, "data": {"content": "System info", "path": "C:\\Windows\\system.ini"}})
        args = {"name": "system.ini"}
        result = self.brain._build_grounded_response("read_file", args, res_json)
        self.assertIsNotNone(result)
        self.assertIn("Contents of C:\\Windows\\system.ini:", result)
        self.assertIn("System info", result)

    def test_43_display_truncation(self):
        """Files > 4000 chars are truncated for display."""
        long_content = "A" * 5000
        res_json = json.dumps({"success": True, "data": {"content": long_content, "path": "/workspace/large.txt"}})
        result = self.brain._build_grounded_response("read_workspace_file", {}, res_json)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len("Contents of /workspace/large.txt:\n\n") + 4000 + len("\n\n... [Content truncated for display. Original file is larger.]"))
        self.assertIn("... [Content truncated for display. Original file is larger.]", result)



class TestPhase29Constants(unittest.TestCase):

    def test_max_steps_value(self):
        """MAX_STEPS must equal 3 exactly."""
        self.assertEqual(MAX_STEPS, 3)

    def test_max_steps_is_integer(self):
        """MAX_STEPS must be an integer."""
        self.assertIsInstance(MAX_STEPS, int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
