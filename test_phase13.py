"""
test_phase13.py
===============
Phase 13 -- Screenshot Capture Tests.

Tests cover:
  - take_screenshot: success (file created, filepath returned), directory creation,
    exception handling
  - Tool registry: tool registered, NOT marked dangerous
  - Brain grounding: correct deterministic routing on success/failure
  - Integration: Brain pipeline correctly executes, deterministic confirm string returned,
    LLM bypassed on success
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock, call

from skills import system_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers
# ===========================================================================

_FAKE_PATH = r"C:\Users\91901\Desktop\JARVIS_Screenshots\screenshot_20230101_120000.png"


def _ok_result():
    return json.dumps({
        "success": True,
        "tool": "take_screenshot",
        "data": {"filepath": _FAKE_PATH},
        "error": None,
    })


def _fail_result(error="pyautogui error"):
    return json.dumps({
        "success": False,
        "tool": "take_screenshot",
        "data": None,
        "error": error,
    })


# ===========================================================================
# Unit Tests: take_screenshot
# ===========================================================================

class TestTakeScreenshot:

    @patch("pyautogui.screenshot")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_screenshot_success_dir_exists(self, mock_exists, mock_makedirs, mock_ss):
        """When dir already exists, makedirs not called; screenshot is captured."""
        res = system_ops.take_screenshot()

        assert res["success"] is True
        assert res["tool"] == "take_screenshot"
        assert "filepath" in res["data"]
        assert res["data"]["filepath"].endswith(".png")
        assert "JARVIS_Screenshots" in res["data"]["filepath"]
        # Directory already existed, so makedirs should NOT be called
        mock_makedirs.assert_not_called()
        mock_ss.assert_called_once()

    @patch("pyautogui.screenshot")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_screenshot_creates_dir_if_missing(self, mock_exists, mock_makedirs, mock_ss):
        """When dir doesn't exist, makedirs is called to create it."""
        res = system_ops.take_screenshot()

        assert res["success"] is True
        mock_makedirs.assert_called_once()
        # Confirm the path passed to makedirs contains our safe directory
        call_arg = mock_makedirs.call_args[0][0]
        assert "JARVIS_Screenshots" in call_arg

    @patch("pyautogui.screenshot")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_screenshot_filename_contains_timestamp(self, mock_exists, mock_makedirs, mock_ss):
        """Filename must follow screenshot_YYYYMMDD_HHMMSS.png format."""
        res = system_ops.take_screenshot()
        filename = os.path.basename(res["data"]["filepath"])
        # Must match the pattern: screenshot_ followed by 15 digits and .png
        import re
        assert re.match(r"screenshot_\d{8}_\d{6}\.png", filename), \
            f"Unexpected filename: {filename}"

    @patch("pyautogui.screenshot")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_screenshot_pyautogui_receives_filepath(self, mock_exists, mock_makedirs, mock_ss):
        """pyautogui.screenshot must be called with the exact filepath from our generator."""
        res = system_ops.take_screenshot()
        filepath = res["data"]["filepath"]
        mock_ss.assert_called_once_with(filepath)

    @patch("pyautogui.screenshot", side_effect=Exception("Screen capture failed"))
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_screenshot_exception_caught(self, mock_exists, mock_makedirs, mock_ss):
        """An exception from pyautogui must be caught and returned as failure."""
        res = system_ops.take_screenshot()

        assert res["success"] is False
        assert res["tool"] == "take_screenshot"
        assert "Screen capture failed" in res["error"]


# ===========================================================================
# Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_screenshot_tool_registered(self):
        registry = ToolRegistry()
        assert "take_screenshot" in registry.tools

    def test_screenshot_tool_not_dangerous(self):
        """Screenshot is a safe, non-destructive action. Must NOT require confirmation."""
        registry = ToolRegistry()
        _, is_dangerous = registry.tools["take_screenshot"]
        assert is_dangerous is False


# ===========================================================================
# Brain Grounding Tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_success_returns_deterministic_string(self):
        grounded = self.brain._build_grounded_response("take_screenshot", {}, _ok_result())
        assert grounded is not None
        assert "Done! Screenshot saved to" in grounded
        assert "JARVIS_Screenshots" in grounded

    def test_grounding_success_shows_filename(self):
        grounded = self.brain._build_grounded_response("take_screenshot", {}, _ok_result())
        # Should show the filename in the response
        assert "screenshot_20230101_120000.png" in grounded

    def test_grounding_failure_returns_deterministic_error(self):
        grounded = self.brain._build_grounded_response(
            "take_screenshot", {}, _fail_result("pyautogui error"))
        assert grounded is not None
        assert "failed" in grounded.lower()

    def test_grounding_success_does_not_return_none(self):
        """Ensure success does NOT route to LLM Pass-2 (should be deterministic)."""
        grounded = self.brain._build_grounded_response("take_screenshot", {}, _ok_result())
        assert grounded is not None


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="Done."):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value=llm_final)
        return brain

    def test_take_screenshot_full_pipeline(self):
        brain = self._make_brain('{"tool": "take_screenshot", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_ok_result())

        response = brain.process_input("Take a screenshot")

        brain.registry.execute_tool.assert_called_once_with(
            name="take_screenshot", args={}, confirm_callback=brain._ask_confirmation)
        # LLM must be bypassed on success
        brain.llm.chat.assert_not_called()
        assert "Done! Screenshot saved to" in response
        assert "JARVIS_Screenshots" in response

    def test_take_screenshot_failure_pipeline(self):
        brain = self._make_brain('{"tool": "take_screenshot", "args": {}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_fail_result("pyautogui error"))

        response = brain.process_input("Capture the screen")

        assert "failed" in response.lower()
        brain.llm.chat.assert_not_called()
