"""
test_phase11.py
===============
Phase 11 -- Secure Screen Brightness Control Tests.

Tests cover:
  - get_brightness: success, no levels returned, exception
  - set_brightness: success (0, 50, 100), negative, >100, float, string, None, bool, exception
  - Tool registry: tools registered, none marked dangerous
  - Brain grounding: correct routing (deterministic vs Pass-2)
  - Integration: full Brain.process_input() pipeline
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from skills import system_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers
# ===========================================================================

def _bright_result(level):
    return json.dumps({"success": True, "tool": "get_brightness",
                        "data": {"level": level}, "error": None})

def _set_bright_result(level):
    return json.dumps({"success": True, "tool": "set_brightness",
                        "data": {"level": level}, "error": None})

def _failure_result(tool, error):
    return json.dumps({"success": False, "tool": tool,
                        "data": None, "error": error})

# ===========================================================================
# get_brightness unit tests
# ===========================================================================

class TestGetBrightness:

    @patch("skills.system_ops.sbc", create=True)
    def test_get_brightness_success(self, mock_sbc):
        mock_sbc.get_brightness.return_value = [75, 100]
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.get_brightness()
        
        assert res["success"] is True
        assert res["tool"] == "get_brightness"
        assert res["data"]["level"] == 75
        assert res["error"] is None

    @patch("skills.system_ops.sbc", create=True)
    def test_get_brightness_no_levels(self, mock_sbc):
        mock_sbc.get_brightness.return_value = []
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.get_brightness()
        
        assert res["success"] is False
        assert "No display brightness levels could be read" in res["error"]

    @patch("skills.system_ops.sbc", create=True)
    def test_get_brightness_exception(self, mock_sbc):
        mock_sbc.get_brightness.side_effect = Exception("WMI Error")
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.get_brightness()
            
        assert res["success"] is False
        assert "WMI Error" in res["error"]

# ===========================================================================
# set_brightness unit tests
# ===========================================================================

class TestSetBrightness:

    @patch("skills.system_ops.sbc", create=True)
    def test_set_brightness_success_50(self, mock_sbc):
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.set_brightness(50)
        
        assert res["success"] is True
        assert res["tool"] == "set_brightness"
        assert res["data"]["level"] == 50
        mock_sbc.set_brightness.assert_called_once_with(50, display=0)

    @patch("skills.system_ops.sbc", create=True)
    def test_set_brightness_success_0(self, mock_sbc):
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.set_brightness(0)
        assert res["success"] is True
        assert res["data"]["level"] == 0

    @patch("skills.system_ops.sbc", create=True)
    def test_set_brightness_success_100(self, mock_sbc):
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.set_brightness(100)
        assert res["success"] is True
        assert res["data"]["level"] == 100

    def test_set_brightness_negative(self):
        res = system_ops.set_brightness(-10)
        assert res["success"] is False
        assert "between 0 and 100" in res["error"]

    def test_set_brightness_over_100(self):
        res = system_ops.set_brightness(110)
        assert res["success"] is False
        assert "between 0 and 100" in res["error"]

    def test_set_brightness_float(self):
        res = system_ops.set_brightness(50.5)
        assert res["success"] is False
        assert "integer" in res["error"].lower()

    def test_set_brightness_string(self):
        res = system_ops.set_brightness("50")
        assert res["success"] is False
        assert "integer" in res["error"].lower()

    def test_set_brightness_none(self):
        res = system_ops.set_brightness(None)
        assert res["success"] is False

    def test_set_brightness_bool_rejected(self):
        assert system_ops.set_brightness(True)["success"] is False
        assert system_ops.set_brightness(False)["success"] is False

    @patch("skills.system_ops.sbc", create=True)
    def test_set_brightness_exception(self, mock_sbc):
        mock_sbc.set_brightness.side_effect = Exception("Display not found")
        with patch.dict("sys.modules", {"screen_brightness_control": mock_sbc}):
            res = system_ops.set_brightness(75)
        assert res["success"] is False
        assert "Display not found" in res["error"]

# ===========================================================================
# Tool registry tests
# ===========================================================================

class TestToolRegistry:

    def test_brightness_tools_registered(self):
        registry = ToolRegistry()
        assert "get_brightness" in registry.tools
        assert "set_brightness" in registry.tools

    def test_brightness_tools_not_dangerous(self):
        registry = ToolRegistry()
        for name in ("get_brightness", "set_brightness"):
            _, dangerous = registry.tools[name]
            assert dangerous is False

# ===========================================================================
# Brain grounding tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_get_brightness_success_none(self):
        grounded = self.brain._build_grounded_response("get_brightness", {}, _bright_result(70))
        assert grounded is None

    def test_grounding_get_brightness_failure(self):
        grounded = self.brain._build_grounded_response(
            "get_brightness", {}, _failure_result("get_brightness", "WMI failure"))
        assert grounded is not None
        assert "brightness" in grounded.lower()
        assert "failed" in grounded.lower()

    def test_grounding_set_brightness_success(self):
        grounded = self.brain._build_grounded_response(
            "set_brightness", {"level": 80}, _set_bright_result(80))
        assert grounded is not None
        assert "80%" in grounded

    def test_grounding_set_brightness_failure(self):
        grounded = self.brain._build_grounded_response(
            "set_brightness", {"level": -1},
            _failure_result("set_brightness", "Brightness level must be an integer between 0 and 100."))
        assert grounded is not None
        assert "failed" in grounded.lower()

# ===========================================================================
# Integration tests
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="Brightness noted."):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value=llm_final)
        return brain

    def test_set_brightness_pipeline(self):
        brain = self._make_brain('{"tool": "set_brightness", "args": {"level": 60}}')
        brain.registry.execute_tool = MagicMock(return_value=_set_bright_result(60))
        response = brain.process_input("Set the brightness to 60")
        brain.registry.execute_tool.assert_called_once_with(
            name="set_brightness", args={"level": 60}, confirm_callback=brain._ask_confirmation)
        assert "60%" in response
        brain.llm.chat.assert_not_called()

    def test_get_brightness_pipeline(self):
        brain = self._make_brain('{"tool": "get_brightness", "args": {}}',
                                 llm_final="The current brightness is 55%.")
        brain.registry.execute_tool = MagicMock(return_value=_bright_result(55))
        response = brain.process_input("What is the current brightness?")
        brain.registry.execute_tool.assert_called_once_with(
            name="get_brightness", args={}, confirm_callback=brain._ask_confirmation)
        brain.llm.chat.assert_called_once()
        assert "55%" in response

    def test_set_brightness_failure_pipeline(self):
        brain = self._make_brain('{"tool": "set_brightness", "args": {"level": 150}}')
        brain.registry.execute_tool = MagicMock(
            return_value=_failure_result("set_brightness",
                "Brightness level must be an integer between 0 and 100."))
        response = brain.process_input("Set brightness to 150")
        assert "failed" in response.lower()
        brain.llm.chat.assert_not_called()
