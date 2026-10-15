"""
test_phase12.py
===============
Phase 12 -- Secure System Power Operations Tests.

Tests cover:
  - lock_screen: success, exception, non-Windows platform rejection
  - sleep_system: success, exception, non-Windows platform rejection
  - Tool registry: tools registered AND marked dangerous
  - Brain grounding: correct deterministic routing for success/failure
  - Integration: Brain pipeline properly halts for user confirmation on these tools
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

def _lock_result():
    return json.dumps({"success": True, "tool": "lock_screen", "data": {}, "error": None})

def _sleep_result():
    return json.dumps({"success": True, "tool": "sleep_system", "data": {}, "error": None})

def _failure_result(tool, error):
    return json.dumps({"success": False, "tool": tool, "data": None, "error": error})


# ===========================================================================
# Unit Tests: lock_screen
# ===========================================================================

class TestLockScreen:

    @patch("sys.platform", "win32")
    def test_lock_screen_success(self):
        with patch("ctypes.windll.user32.LockWorkStation", return_value=1) as mock_lock:
            res = system_ops.lock_screen()
            
        assert res["success"] is True
        assert res["tool"] == "lock_screen"
        mock_lock.assert_called_once()

    @patch("sys.platform", "win32")
    def test_lock_screen_failure(self):
        with patch("ctypes.windll.user32.LockWorkStation", return_value=0):
            res = system_ops.lock_screen()
            
        assert res["success"] is False
        assert "failed" in res["error"]

    @patch("sys.platform", "win32")
    def test_lock_screen_exception(self):
        with patch("ctypes.windll.user32.LockWorkStation", side_effect=Exception("API Error")):
            res = system_ops.lock_screen()
            
        assert res["success"] is False
        assert "API Error" in res["error"]

    @patch("sys.platform", "linux")
    def test_lock_screen_not_windows(self):
        res = system_ops.lock_screen()
        assert res["success"] is False
        assert "only supported on Windows" in res["error"]


# ===========================================================================
# Unit Tests: sleep_system
# ===========================================================================

class TestSleepSystem:

    @patch("sys.platform", "win32")
    def test_sleep_system_success(self):
        with patch("ctypes.windll.powrprof.SetSuspendState", return_value=1) as mock_sleep:
            res = system_ops.sleep_system()
            
        assert res["success"] is True
        assert res["tool"] == "sleep_system"
        # Must be called with (0, 1, 0)
        mock_sleep.assert_called_once_with(0, 1, 0)

    @patch("sys.platform", "win32")
    def test_sleep_system_exception(self):
        with patch("ctypes.windll.powrprof.SetSuspendState", side_effect=Exception("Power API Error")):
            res = system_ops.sleep_system()
            
        assert res["success"] is False
        assert "Power API Error" in res["error"]

    @patch("sys.platform", "darwin")
    def test_sleep_system_not_windows(self):
        res = system_ops.sleep_system()
        assert res["success"] is False
        assert "only supported on Windows" in res["error"]


# ===========================================================================
# Tool Registry Tests
# ===========================================================================

class TestToolRegistry:

    def test_power_tools_registered(self):
        registry = ToolRegistry()
        assert "lock_screen" in registry.tools
        assert "sleep_system" in registry.tools

    def test_power_tools_ARE_DANGEROUS(self):
        """CRITICAL: These tools must require user confirmation."""
        registry = ToolRegistry()
        _, lock_dangerous = registry.tools["lock_screen"]
        _, sleep_dangerous = registry.tools["sleep_system"]
        assert lock_dangerous is True
        assert sleep_dangerous is True


# ===========================================================================
# Brain Grounding Tests
# ===========================================================================

class TestBrainGrounding:

    def setup_method(self):
        self.brain = Brain()

    def test_grounding_lock_screen_success(self):
        grounded = self.brain._build_grounded_response("lock_screen", {}, _lock_result())
        assert grounded == "Screen locked."

    def test_grounding_sleep_system_success(self):
        grounded = self.brain._build_grounded_response("sleep_system", {}, _sleep_result())
        assert grounded == "System put to sleep."

    def test_grounding_lock_screen_failure(self):
        grounded = self.brain._build_grounded_response(
            "lock_screen", {}, _failure_result("lock_screen", "API failed"))
        assert grounded is not None
        assert "failed" in grounded.lower()

    def test_grounding_sleep_system_failure(self):
        grounded = self.brain._build_grounded_response(
            "sleep_system", {}, _failure_result("sleep_system", "API failed"))
        assert grounded is not None
        assert "failed" in grounded.lower()


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestBrainIntegration:

    def _make_brain(self, intent_json, llm_final="Done"):
        brain = Brain()
        brain.llm.chat_json = MagicMock(return_value=intent_json)
        brain.llm.chat      = MagicMock(return_value=llm_final)
        return brain

    def test_lock_screen_pipeline(self):
        brain = self._make_brain('{"tool": "lock_screen", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_lock_result())
        
        response = brain.process_input("Lock my screen")
        
        # Verify the brain passed its own confirmation callback
        brain.registry.execute_tool.assert_called_once_with(
            name="lock_screen", args={}, confirm_callback=brain._ask_confirmation)
        
        assert response == "Screen locked."
        brain.llm.chat.assert_not_called()

    def test_sleep_system_pipeline(self):
        brain = self._make_brain('{"tool": "sleep_system", "args": {}}')
        brain.registry.execute_tool = MagicMock(return_value=_sleep_result())
        
        response = brain.process_input("Put the computer to sleep")
        
        brain.registry.execute_tool.assert_called_once_with(
            name="sleep_system", args={}, confirm_callback=brain._ask_confirmation)
            
        assert response == "System put to sleep."
        brain.llm.chat.assert_not_called()

    @patch("builtins.input", return_value="n")
    def test_registry_actually_halts_for_dangerous_tools(self, mock_input):
        """Proves that because is_dangerous=True, the registry actually halts for confirmation."""
        registry = ToolRegistry()
        # Mock the underlying skill so we don't actually sleep the test runner if something breaks
        with patch("skills.system_ops.sleep_system") as mock_sleep:
            # Execute with user pressing 'n' (No) - lambda returns False
            result_str = registry.execute_tool("sleep_system", {}, confirm_callback=lambda name, args: False)
            
            assert "User denied permission" in result_str
            mock_sleep.assert_not_called()
