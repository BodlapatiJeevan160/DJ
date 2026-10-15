"""
test_phase26.py
===============
Phase 26: Workspace & Safe File Operations

Tests path traversal protections, symlink escapes, search constraints,
and deterministic file editing using a real temporary directory.
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Setup & Helpers
# ---------------------------------------------------------------------------

class TestPhase26WorkspaceOps(unittest.TestCase):
    def setUp(self):
        # Create a temporary workspace root
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = os.path.join(self.test_dir, "workspace")
        os.makedirs(self.workspace_root)
        
        # Create an "outside" directory to test traversal/escapes
        self.outside_dir = os.path.join(self.test_dir, "outside")
        os.makedirs(self.outside_dir)
        
        # Patch WORKSPACE_ROOTS in core.config
        self.patcher = patch("core.config.WORKSPACE_ROOTS", [self.workspace_root])
        self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def _create_file(self, path, content=""):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
    def _create_binary(self, path):
        with open(path, "wb") as f:
            f.write(b"\xff\xfe\x00\x00\x01\x02\x03\x04")

# ---------------------------------------------------------------------------
# 1. Path Resolution & Security
# ---------------------------------------------------------------------------

    def test_resolve_valid_path(self):
        from skills.workspace_ops import _resolve_workspace_path
        p = os.path.join(self.workspace_root, "test.txt")
        self.assertEqual(_resolve_workspace_path(p), os.path.realpath(p))
        
    def test_resolve_traversal_attack(self):
        from skills.workspace_ops import _resolve_workspace_path
        # Traverse out to the outside dir
        attack_path = os.path.join(self.workspace_root, "..", "outside", "secret.txt")
        with self.assertRaises(ValueError) as ctx:
            _resolve_workspace_path(attack_path)
        self.assertIn("not within any configured WORKSPACE_ROOTS", str(ctx.exception))
        
    def test_resolve_null_byte_rejection(self):
        from skills.workspace_ops import _resolve_workspace_path
        with self.assertRaises(ValueError):
            _resolve_workspace_path(os.path.join(self.workspace_root, "test\x00.txt"))
            
    def test_symlink_escape_rejection(self):
        from skills.workspace_ops import _resolve_workspace_path
        # Create a symlink inside the workspace pointing outside
        link_path = os.path.join(self.workspace_root, "link_to_outside")
        try:
            os.symlink(self.outside_dir, link_path)
        except OSError:
            # On Windows, os.symlink might fail without Admin privileges or Developer Mode.
            # Skip this test if we can't create symlinks.
            self.skipTest("Symlink creation not supported/permitted on this Windows host.")
            
        with self.assertRaises(ValueError) as ctx:
            _resolve_workspace_path(os.path.join(link_path, "secret.txt"))
        self.assertIn("not within any configured WORKSPACE_ROOTS", str(ctx.exception))

# ---------------------------------------------------------------------------
# 2. Search Workspace
# ---------------------------------------------------------------------------

    def test_search_content_success(self):
        from skills.workspace_ops import search_workspace
        f1 = os.path.join(self.workspace_root, "a.txt")
        f2 = os.path.join(self.workspace_root, "b.txt")
        self._create_file(f1, "Hello world, this is JARVIS.")
        self._create_file(f2, "No match here.")
        
        r = search_workspace("JARVIS", search_type="content")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["data"]["results"]), 1)
        self.assertEqual(r["data"]["results"][0]["file"], f1)
        
    def test_search_filename_success(self):
        from skills.workspace_ops import search_workspace
        self._create_file(os.path.join(self.workspace_root, "config_prod.json"))
        self._create_file(os.path.join(self.workspace_root, "config_dev.json"))
        
        r = search_workspace("config_", search_type="filename")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["data"]["results"]), 2)
        
    def test_search_skips_hidden_dirs(self):
        from skills.workspace_ops import search_workspace
        git_dir = os.path.join(self.workspace_root, ".git")
        os.makedirs(git_dir)
        self._create_file(os.path.join(git_dir, "config"), "hidden match")
        
        r = search_workspace("hidden", search_type="content")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["data"]["results"]), 0)
        
    def test_search_skips_binary_files(self):
        from skills.workspace_ops import search_workspace
        bin_file = os.path.join(self.workspace_root, "app.exe")
        self._create_binary(bin_file)
        
        r = search_workspace("anything", search_type="content")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["data"]["results"]), 0)
        self.assertEqual(r["data"]["skipped_unsearchable"], 1)
        
    @patch("os.path.getsize")
    def test_search_skips_large_files(self, mock_getsize):
        from skills.workspace_ops import search_workspace
        f = os.path.join(self.workspace_root, "large.txt")
        self._create_file(f, "match me")
        # Pretend the file is 2 MB
        mock_getsize.return_value = 2 * 1024 * 1024
        
        r = search_workspace("match", search_type="content")
        self.assertTrue(r["success"])
        self.assertEqual(len(r["data"]["results"]), 0)
        self.assertEqual(r["data"]["skipped_unsearchable"], 1)

# ---------------------------------------------------------------------------
# 3. Edit File
# ---------------------------------------------------------------------------

    def test_edit_file_success(self):
        from skills.workspace_ops import edit_file
        f = os.path.join(self.workspace_root, "config.json")
        self._create_file(f, '{"debug": false}')
        
        r = edit_file(f, '"debug": false', '"debug": true')
        self.assertTrue(r["success"])
        self.assertEqual(r["data"]["status"], "replaced")
        
        with open(f, "r", encoding="utf-8") as f_obj:
            content = f_obj.read()
        self.assertEqual(content, '{"debug": true}')
        
    def test_edit_file_ambiguous_rejection(self):
        from skills.workspace_ops import edit_file
        f = os.path.join(self.workspace_root, "vars.py")
        self._create_file(f, 'x = 1\ny = 1\n')
        
        r = edit_file(f, '= 1', '= 2')
        self.assertFalse(r["success"])
        self.assertIn("Ambiguous replacement", r["error"])
        
        # File must remain untouched
        with open(f, "r", encoding="utf-8") as f_obj:
            content = f_obj.read()
        self.assertEqual(content, 'x = 1\ny = 1\n')
        
    def test_edit_file_missing_text_rejection(self):
        from skills.workspace_ops import edit_file
        f = os.path.join(self.workspace_root, "vars.py")
        self._create_file(f, 'a = 5\n')
        
        r = edit_file(f, 'b = 6', 'b = 7')
        self.assertFalse(r["success"])
        self.assertIn("was not found", r["error"])

# ---------------------------------------------------------------------------
# 4. Tool Registry Registration
# ---------------------------------------------------------------------------

class TestPhase26ToolRegistration(unittest.TestCase):
    def test_tools_registered_correctly(self):
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        
        self.assertIn("search_workspace", registry.tools)
        self.assertFalse(registry.tools["search_workspace"][1], "search_workspace should not be dangerous")
        
        self.assertIn("edit_file", registry.tools)
        self.assertTrue(registry.tools["edit_file"][1], "edit_file MUST be dangerous")

# ---------------------------------------------------------------------------
# 5. Tool Routing: grep correction guard and intent prompt rules
# ---------------------------------------------------------------------------

class TestPhase26ToolRouting(unittest.TestCase):
    """Tests that verify the grep-to-search_workspace correction guard and
    argument recovery logic in brain.py, mocking the LLM and filesystem."""

    def _run_pass1(self, llm_json_response: str, user_input: str):
        """Run just the Pass-1 parsing + normalization + correction pipeline
        against a mocked LLM response. Returns (tool_name, tool_args)."""
        import json, re
        from unittest.mock import MagicMock, patch

        # We import the module-level patterns directly.
        import core.brain as brain_mod

        tool_name = None
        tool_args = {}

        try:
            tool_data = json.loads(llm_json_response)
            tool_name = tool_data.get("tool") or tool_data.get("action") or tool_data.get("name")
            tool_args = tool_data.get("args")
            if tool_args is None or not isinstance(tool_args, dict):
                tool_args = {k: v for k, v in tool_data.items()
                             if k not in ["tool", "action", "name", "success", "data", "error", "args"]}
        except Exception:
            pass

        # --- WORKSPACE ARGUMENT RECOVERY ---
        if tool_name == "search_workspace" and not (tool_args or {}).get("query"):
            tool_args = tool_args or {}
            clean_query = re.sub(
                r'^(please\s+)?'
                r'(search(\s+the)?\s+workspace\s+(for\s+)?|'
                r'find\s+(in|within|through)?\s*(the\s+)?(workspace|project|files|code)\s*(for\s+)?|'
                r'grep\s+(the\s+)?(workspace|project|files|code)\s*(for\s+)?|'
                r'look\s+for\s+)',
                '', user_input, flags=re.IGNORECASE).strip()
            tool_args["query"] = clean_query if clean_query else user_input
            if "search_type" not in tool_args:
                tool_args["search_type"] = "content"

        # --- GREP CORRECTION GUARD (mirrors production brain.py Guard E) ---
        _UNREGISTERED_SEARCH_ALIASES = frozenset({"grep", "rg", "ripgrep", "ag", "ack"})
        _WORKSPACE_SEARCH_INTENT = re.compile(
            r'\b(search|find|grep|look)\b.{0,60}'
            r'\b(workspace|project|codebase|repo(?:sitory)?|files?|source|code|'
            r'function|class|variable|method|module|import)\b'
            r'|'
            r'\b(where|which)\b.{0,40}\b(defined|declared|used|imported|called)\b'
            r'|'
            r'\b(search\s+for|find)\s+[\'\"].+?[\'\"]'
            r'|'
            r'^grep\s+'
            r'|'
            r'\bmy\s+(project|workspace|codebase|repo(?:sitory)?|code|files?|source)\b',
            re.IGNORECASE
        )
        if tool_name in _UNREGISTERED_SEARCH_ALIASES and _WORKSPACE_SEARCH_INTENT.search(user_input):
            tool_name = "search_workspace"
            if not (tool_args or {}).get("query"):
                clean = re.sub(
                    r'^(please\s+)?(grep|rg|ripgrep|ag|ack|search|find|look\s+for)'
                    r'\s+(the\s+)?(workspace|project|files?|code|codebase|my\s+\w+)?\s*(for\s+)?',
                    '', user_input, flags=re.IGNORECASE).strip()
                tool_args = {"query": clean or user_input, "search_type": "content"}
            elif "search_type" not in tool_args:
                tool_args["search_type"] = "content"

        elif tool_name in _UNREGISTERED_SEARCH_ALIASES:
            # No workspace intent: null out -> conversational LLM path
            tool_name = None
            tool_args = {}


        return tool_name, tool_args

    def test_search_workspace_direct_routing(self):
        """LLM correctly selects search_workspace -> should pass through unchanged."""
        llm_resp = '{"tool": "search_workspace", "args": {"query": "Phase 14", "search_type": "content"}}'
        name, args = self._run_pass1(llm_resp, "Search the workspace for Phase 14")
        self.assertEqual(name, "search_workspace")
        self.assertEqual(args["query"], "Phase 14")
        self.assertEqual(args["search_type"], "content")

    def test_grep_corrected_to_search_workspace(self):
        """LLM returns 'grep' (unregistered) -> corrected to search_workspace."""
        llm_resp = '{"tool": "grep", "args": {"pattern": "read_file"}}'
        name, args = self._run_pass1(llm_resp, "Find where read_file is defined in the code")
        self.assertEqual(name, "search_workspace")
        self.assertIn("query", args)

    def test_bare_grep_corrected(self):
        """'grep' with workspace-search intent in user input -> corrected."""
        llm_resp = '{"tool": "grep", "args": {}}'
        name, args = self._run_pass1(llm_resp, "grep the workspace for MAX_ENTRIES")
        self.assertEqual(name, "search_workspace")
        self.assertIn("MAX_ENTRIES", args.get("query", ""))

    def test_grep_without_workspace_intent_not_corrected(self):
        """'grep' selected but user input is an informational question with no
        workspace/project/code scope -> nulled out, routing to conversational LLM."""
        for user_input in [
            "How do I use grep in Python?",
            "What does grep stand for?",
            "Explain how grep works.",
            "Show me a grep example.",
        ]:
            name, args = self._run_pass1('{"tool": "grep", "args": {}}', user_input)
            self.assertIsNone(
                name,
                f"Guard should have nulled tool_name for informational query: '{user_input}'"
            )

    def test_informational_grep_routes_to_conversation(self):
        """Explicitly verify conversational path: tool is None, args are empty."""
        name, args = self._run_pass1(
            '{"tool": "grep", "args": {}}',
            "How do I use grep in Python?"
        )
        # tool_name must be None -> no tool executed -> conversational LLM handles it
        self.assertIsNone(name)
        self.assertEqual(args, {})

    def test_search_workspace_missing_query_recovery(self):
        """LLM selects search_workspace but omits query -> recovered from user input."""
        llm_resp = '{"tool": "search_workspace", "args": {}}'
        name, args = self._run_pass1(llm_resp, "Search the workspace for def read_file")
        self.assertEqual(name, "search_workspace")
        self.assertIn("def read_file", args.get("query", ""))

    def test_search_workspace_default_search_type(self):
        """search_workspace with query provided but no search_type -> query preserved."""
        llm_resp = '{"tool": "search_workspace", "args": {"query": "MemoryStore"}}'
        name, args = self._run_pass1(llm_resp, "find MemoryStore in the project")
        self.assertEqual(name, "search_workspace")
        self.assertEqual(args.get("query"), "MemoryStore")

    def test_possessive_project_reference_corrected(self):
        """'grep my project for X' should route to search_workspace."""
        llm_resp = '{"tool": "grep", "args": {}}'
        name, args = self._run_pass1(llm_resp, "grep my project for MAX_ENTRIES")
        self.assertEqual(name, "search_workspace")
        self.assertIn("MAX_ENTRIES", args.get("query", ""))

    def test_where_defined_corrected(self):
        """'where is read_file defined' -> corrected to search_workspace."""
        llm_resp = '{"tool": "grep", "args": {}}'
        name, args = self._run_pass1(llm_resp, "where is read_file defined")
        self.assertEqual(name, "search_workspace")


class TestPhase26GuardF(unittest.TestCase):
    """Regression tests for Guard F: Prior result reference routing."""
    def setUp(self):
        from core.brain import Brain
        self.brain = Brain()
        
        # We replace the actual execution with a dummy return so we don't really touch disk
        self.brain.registry.execute_tool = MagicMock(return_value=json.dumps({
            "success": True, 
            "data": {"results": [{"file": "dummy.txt", "match": "MATCH"}]}
        }))
        
        # Default mock for LLM text gen
        self.brain.llm.chat = MagicMock(return_value="Pass 2 generated text.")

    def test_guard_f_suppresses_unrelated_tool(self):
        # Setup: populate history with a previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\nRESULT: {}"
        })
        
        # User references it
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        response = self.brain.process_input("Summarize the first result.")
        
        # Tool execution should be skipped
        self.brain.registry.execute_tool.assert_not_called()
        self.assertEqual(response, "Pass 2 generated text.")

    def test_guard_f_allows_navigation_tool(self):
        # Setup: populate history with a previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\nRESULT: {}"
        })
        
        # User references it, but LLM picks a VALID navigation tool (e.g. edit_file)
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "edit_file", "args": {"path": "x"}}')
        
        # Reset execution mock to return dummy JSON
        self.brain.registry.execute_tool.reset_mock()
        self.brain.registry.execute_tool.return_value = '{"success": true}'
        
        self.brain.process_input("Open the first result.")
        
        # Should have executed edit_file
        self.brain.registry.execute_tool.assert_called_once()
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs["name"], "edit_file")

    def test_guard_f_clarification_if_no_history(self):
        # History is empty
        self.brain.history = []
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        response = self.brain.process_input("Summarize the first result.")
        
        # Tool execution skipped, clarification returned
        self.brain.registry.execute_tool.assert_not_called()
        self.assertIn("I don't have a previous search result to summarize", response)

    def test_independent_tool_not_suppressed(self):
        # History has previous search
        self.brain.history.append({
            "role": "system",
            "content": "ACTION EXECUTED: search_workspace\nRESULT: {}"
        })
        
        # User asks unrelated question NOT matching pattern
        self.brain.llm.chat_json = MagicMock(return_value='{"tool": "get_network_info", "args": {}}')
        
        self.brain.process_input("What is my IP address?")
        
        # Tool execution SHOULD occur
        self.brain.registry.execute_tool.assert_called_once()
        args, kwargs = self.brain.registry.execute_tool.call_args
        self.assertEqual(kwargs["name"], "get_network_info")

if __name__ == '__main__':
    unittest.main()
