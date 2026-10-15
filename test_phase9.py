"""
test_phase9.py
==============
Phase 9 — Secure Clipboard Operations Unit Tests.

Tests cover:
  - get_clipboard: success, empty clipboard
  - set_clipboard: success, too large, null bytes
  - Tool registry: both tools registered
  - Brain grounding: correct deterministic vs Pass-2 routing
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from skills import system_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain


# ---------------------------------------------------------------------------
# get_clipboard tests
# ---------------------------------------------------------------------------

def test_get_clipboard_success():
    with patch("skills.system_ops.pyperclip.paste", return_value="Hello JARVIS"):
        res = system_ops.get_clipboard()
    assert res["success"] is True
    assert res["tool"] == "get_clipboard"
    assert res["data"]["text"] == "Hello JARVIS"
    assert res["error"] is None


def test_get_clipboard_empty():
    with patch("skills.system_ops.pyperclip.paste", return_value=""):
        res = system_ops.get_clipboard()
    assert res["success"] is False
    assert "empty" in res["error"].lower()


def test_get_clipboard_none():
    with patch("skills.system_ops.pyperclip.paste", return_value=None):
        res = system_ops.get_clipboard()
    assert res["success"] is False
    assert "empty" in res["error"].lower()


def test_get_clipboard_exception():
    with patch("skills.system_ops.pyperclip.paste", side_effect=Exception("no clipboard")):
        res = system_ops.get_clipboard()
    assert res["success"] is False
    assert "no clipboard" in res["error"]


# ---------------------------------------------------------------------------
# set_clipboard tests
# ---------------------------------------------------------------------------

def test_set_clipboard_success():
    with patch("skills.system_ops.pyperclip.copy") as mock_copy:
        res = system_ops.set_clipboard("Copy this text.")
    assert res["success"] is True
    assert res["tool"] == "set_clipboard"
    assert res["data"]["written"] == len("Copy this text.")
    mock_copy.assert_called_once_with("Copy this text.")


def test_set_clipboard_too_large():
    big_text = "A" * (10 * 1024 + 1)
    res = system_ops.set_clipboard(big_text)
    assert res["success"] is False
    assert "too large" in res["error"]


def test_set_clipboard_null_bytes():
    res = system_ops.set_clipboard("Hello\x00World")
    assert res["success"] is False
    assert "null bytes" in res["error"]


def test_set_clipboard_non_string():
    res = system_ops.set_clipboard(12345)  # type: ignore
    assert res["success"] is False
    assert "plain text" in res["error"].lower()


def test_set_clipboard_exception():
    with patch("skills.system_ops.pyperclip.copy", side_effect=Exception("clipboard locked")):
        res = system_ops.set_clipboard("test")
    assert res["success"] is False
    assert "clipboard locked" in res["error"]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def test_tool_registry_clipboard():
    registry = ToolRegistry()
    assert "get_clipboard" in registry.tools
    assert "set_clipboard" in registry.tools
    # Neither should require user confirmation
    _, get_dangerous = registry.tools["get_clipboard"]
    _, set_dangerous = registry.tools["set_clipboard"]
    assert get_dangerous is False
    assert set_dangerous is False


# ---------------------------------------------------------------------------
# Brain grounding
# ---------------------------------------------------------------------------

def test_brain_grounding_get_clipboard_success():
    """get_clipboard success → returns None (routes to LLM Pass 2)."""
    brain = Brain()
    result = json.dumps({"success": True, "tool": "get_clipboard",
                         "data": {"text": "Some copied text"}, "error": None})
    grounded = brain._build_grounded_response("get_clipboard", {}, result)
    assert grounded is None


def test_brain_grounding_get_clipboard_failure():
    """get_clipboard failure → deterministic error string."""
    brain = Brain()
    result = json.dumps({"success": False, "tool": "get_clipboard",
                         "data": None, "error": "Clipboard is empty."})
    grounded = brain._build_grounded_response("get_clipboard", {}, result)
    assert grounded is not None
    assert "clipboard" in grounded.lower()
    assert "failed" in grounded.lower() or "Clipboard is empty" in grounded


def test_brain_grounding_set_clipboard_success():
    """set_clipboard success → deterministic confirmation string."""
    brain = Brain()
    result = json.dumps({"success": True, "tool": "set_clipboard",
                         "data": {"written": 14}, "error": None})
    grounded = brain._build_grounded_response("set_clipboard", {"text": "some copied text"}, result)
    assert grounded is not None
    assert "clipboard" in grounded.lower()
    assert "14" in grounded


def test_brain_grounding_set_clipboard_failure():
    """set_clipboard failure → deterministic error string."""
    brain = Brain()
    result = json.dumps({"success": False, "tool": "set_clipboard",
                         "data": None, "error": "Text is too large to copy to clipboard (max 10KB)."})
    grounded = brain._build_grounded_response("set_clipboard", {"text": "x" * 11000}, result)
    assert grounded is not None
    assert "failed" in grounded.lower()
    assert "too large" in grounded.lower() or "10KB" in grounded
