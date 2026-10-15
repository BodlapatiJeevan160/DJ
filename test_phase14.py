"""
test_phase14.py
===============
Phase 14 -- Secure File Write Tests.

Tests cover:
  - write_file: success on desktop/documents/downloads
  - write_file: content validation (non-string, null bytes, oversized)
  - write_file: path security (traversal, disallowed location)
  - write_file: overwrite existing file
  - write_file: I/O exception handling
  - Tool registry: registered, IS dangerous
  - Brain grounding: deterministic success/failure strings
  - Integration: full Brain pipeline, confirmation callback fires, LLM bypassed
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from skills import file_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers
# ===========================================================================

_FAKE_PATH = r"C:\Users\91901\Desktop\notes.txt"
_CONTENT = "Hello, JARVIS!"
_BYTES = len(_CONTENT.encode("utf-8"))


def _ok_result(path=_FAKE_PATH, bytes_written=_BYTES):
    return json.dumps({
        "success": True,
        "tool": "write_file",
        "data": {"path": path, "bytes_written": bytes_written},
        "error": None,
    })


def _fail_result(error="Permission denied"):
    return json.dumps({
        "success": False,
        "tool": "write_file",
        "data": None,
        "error": error,
    })


# ===========================================================================
# Unit Tests: write_file
# ===========================================================================

class TestWriteFile:

    @patch("builtins.open", mock_open())
    @patch("skills.file_ops._resolve_secure_path", return_value=_FAKE_PATH)
    def test_write_success(self, mock_resolve):
        res = file_ops.write_file("desktop", "notes.txt", _CONTENT)
        assert res["success"] is True
        assert res["tool"] == "write_file"
        assert res["data"]["path"] == _FAKE_PATH
        assert res["data"]["bytes_written"] == _BYTES

    @patch("builtins.open", mock_open())
    @patch("skills.file_ops._resolve_secure_path", return_value=r"C:\Users\91901\Documents\doc.txt")
    def test_write_documents(self, mock_resolve):
        res = file_ops.write_file("documents", "doc.txt", "Some content")
        assert res["success"] is True

    @patch("builtins.open", mock_open())
    @patch("skills.file_ops._resolve_secure_path", return_value=r"C:\Users\91901\Downloads\dl.txt")
    def test_write_downloads(self, mock_resolve):
        res = file_ops.write_file("downloads", "dl.txt", "Download data")
        assert res["success"] is True

    def test_write_non_string_content_int(self):
        res = file_ops.write_file("desktop", "notes.txt", 42)
        assert res["success"] is False
        assert "plain text string" in res["error"]

    def test_write_non_string_content_none(self):
        res = file_ops.write_file("desktop", "notes.txt", None)
        assert res["success"] is False

    def test_write_non_string_content_bool(self):
        res = file_ops.write_file("desktop", "notes.txt", True)
        assert res["success"] is False

    def test_write_null_bytes_rejected(self):
        res = file_ops.write_file("desktop", "notes.txt", "Hello\x00World")
        assert res["success"] is False
        assert "null bytes" in res["error"]

    def test_write_oversized_content_rejected(self):
        big = "A" * (101 * 1024)  # 101 KB, over 100 KB limit
        res = file_ops.write_file("desktop", "notes.txt", big)
        assert res["success"] is False
        assert "too large" in res["error"]

    def test_write_exactly_at_limit_passes(self):
        """Content exactly at 100 KB boundary should pass validation."""
        at_limit = "A" * (100 * 1024)
        # We only check up to content validation; mock path resolution + file open
        with patch("skills.file_ops._resolve_secure_path", return_value=_FAKE_PATH):
            with patch("builtins.open", mock_open()):
                res = file_ops.write_file("desktop", "notes.txt", at_limit)
        assert res["success"] is True

    def test_write_disallowed_location(self):
        res = file_ops.write_file("system32", "evil.txt", "bad content")
        assert res["success"] is False
        assert "not permitted" in res["error"].lower() or "not found" in res["error"].lower() or "location" in res["error"].lower()

    def test_write_path_traversal_rejected(self):
        res = file_ops.write_file("desktop", "../../evil.txt", "bad content")
        assert res["success"] is False

    def test_write_empty_name_rejected(self):
        res = file_ops.write_file("desktop", "   ", _CONTENT)
        assert res["success"] is False

    @patch("skills.file_ops._resolve_secure_path", return_value=_FAKE_PATH)
    def test_write_io_exception_caught(self, mock_resolve):
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            res = file_ops.write_file("desktop", "notes.txt", _CONTENT)
        assert res["success"] is False
        assert "Access denied" in res["error"]

    @patch("builtins.open", mock_open())
    @patch("skills.file_ops._resolve_secure_path", return_value=_FAKE_PATH)
    def test_write_overwrite_existing_succeeds(self, mock_resolve):
        """Overwrite is allowed — the dangerous flag + confirmation protects against accidents."""
        res = file_ops.write_file("desktop", "notes.txt", "New content")
        assert res["success"] is True


# ===========================================================================
# Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_write_file_registered(self):
        registry = ToolRegistry()
        assert "write_file" in registry.tools

    def test_write_file_IS_DANGEROUS(self):
        """CRITICAL: write_file modifies disk state. Must require user confirmation."""
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["write_file"]
        assert is_dangerous is True


# ===========================================================================
# Brain Grounding Tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_success_deterministic(self):
        grounded = self.brain._build_grounded_response("write_file", {}, _ok_result())
        assert grounded is not None
        assert "Done! Wrote" in grounded
        assert str(_BYTES) in grounded

    def test_grounding_success_does_not_return_none(self):
        """Must be deterministic (Pass-1), not routed to LLM (Pass-2)."""
        grounded = self.brain._build_grounded_response("write_file", {}, _ok_result())
        assert grounded is not None

    def test_grounding_failure_deterministic(self):
        grounded = self.brain._build_grounded_response(
            "write_file", {}, _fail_result("Permission denied"))
        assert grounded is not None
        assert "failed" in grounded.lower()


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value="Done.")
        return brain

    def test_write_file_full_pipeline(self):
        brain = self._make_brain(
            '{"tool": "write_file", "args": {"location": "desktop", "name": "notes.txt", "content": "Hello JARVIS"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_ok_result())

        response = brain.process_input("Write Hello JARVIS to notes.txt on my Desktop")

        brain.registry.execute_tool.assert_called_once_with(
            name="write_file",
            args={"location": "desktop", "name": "notes.txt", "content": "Hello JARVIS"},
            confirm_callback=brain._ask_confirmation,
        )
        assert "Done! Wrote" in response
        brain.llm.chat.assert_not_called()

    def test_write_file_failure_pipeline(self):
        brain = self._make_brain(
            '{"tool": "write_file", "args": {"location": "desktop", "name": "notes.txt", "content": "Hello"}}'
        )
        brain.registry.execute_tool = MagicMock(return_value=_fail_result("Permission denied"))

        response = brain.process_input("Save my note to desktop")
        assert "failed" in response.lower()
        brain.llm.chat.assert_not_called()

    def test_registry_confirmation_callback_fires_on_deny(self):
        """Prove the confirmation gate actually blocks execution when denied."""
        registry = ToolRegistry()
        with patch("skills.file_ops.write_file") as mock_write:
            result_str = registry.execute_tool(
                "write_file",
                {"location": "desktop", "name": "notes.txt", "content": "Hello"},
                confirm_callback=lambda name, args: False,  # user denies
            )
            assert "User denied permission" in result_str
            mock_write.assert_not_called()
