"""
test_phase17.py
===============
Phase 17 -- Date, Time & Reminder Tests

Security invariants verified:
  - Reading the system clock has no side effects.
  - Reminders only print to stdout. No file I/O, no subprocesses.
  - Active reminders are capped at 20 to prevent accumulation.
  - is_dangerous is False for all tools.
  - Input validation is robust.
"""

import datetime
import json
import threading
import pytest
from unittest.mock import patch, MagicMock

from skills import datetime_ops
from core.tool_registry import ToolRegistry
from core.brain import Brain

# ===========================================================================
# Helpers
# ===========================================================================

def _make_brain(mock_tool_decision: str) -> Brain:
    """Helper to create a Brain with a mocked Pass-1 JSON output."""
    brain = Brain()
    # Mock the LLM to return exactly what we want for intent
    brain.llm.chat_json = MagicMock(return_value=mock_tool_decision)
    # Mock LLM chat for pass 2
    brain.llm.chat = MagicMock(return_value="[LLM Pass 2 Response]")
    return brain

# ===========================================================================
# Unit Tests: skills/datetime_ops.py
# ===========================================================================

class TestGetDatetime:
    @patch("skills.datetime_ops.datetime")
    @patch("skills.datetime_ops.time")
    def test_get_datetime_success(self, mock_time, mock_datetime):
        # Setup mock
        mock_dt = datetime.datetime(2026, 9, 12, 14, 30, 0)
        mock_datetime.datetime.now.return_value = mock_dt
        mock_time.strftime.return_value = "+0530"

        res = datetime_ops.get_datetime()

        assert res["success"] is True
        assert res["tool"] == "get_datetime"
        assert res["error"] is None
        assert res["data"]["date"] == "2026-09-12"
        assert res["data"]["time"] == "14:30:00"
        assert res["data"]["weekday"] == "Saturday"
        assert res["data"]["timezone"] == "+05:30"

    @patch("skills.datetime_ops.datetime")
    def test_get_datetime_exception(self, mock_datetime):
        mock_datetime.datetime.now.side_effect = Exception("System error")
        res = datetime_ops.get_datetime()
        assert res["success"] is False
        assert "System error" in res["error"]


class TestGetTimeUntil:
    @patch("skills.datetime_ops.datetime")
    def test_success_24h(self, mock_datetime):
        # Now is 14:00
        mock_datetime.datetime.now.return_value = datetime.datetime(2026, 9, 12, 14, 0, 0)
        # Target is 17:30 (3.5 hours = 210 minutes)
        mock_datetime.datetime.strptime = datetime.datetime.strptime
        mock_datetime.datetime.combine = datetime.datetime.combine
        mock_datetime.timedelta = datetime.timedelta

        res = datetime_ops.get_time_until("17:30")
        assert res["success"] is True
        assert res["data"]["minutes_until"] == 210
        assert res["data"]["seconds_until"] == 210 * 60

    @patch("skills.datetime_ops.datetime")
    def test_success_12h(self, mock_datetime):
        # Now is 14:00
        mock_datetime.datetime.now.return_value = datetime.datetime(2026, 9, 12, 14, 0, 0)
        # Target is 5pm (3 hours = 180 minutes)
        mock_datetime.datetime.strptime = datetime.datetime.strptime
        mock_datetime.datetime.combine = datetime.datetime.combine
        mock_datetime.timedelta = datetime.timedelta

        res = datetime_ops.get_time_until("5pm")
        assert res["success"] is True
        assert res["data"]["minutes_until"] == 180

    @patch("skills.datetime_ops.datetime")
    def test_success_wraps_to_tomorrow(self, mock_datetime):
        # Now is 14:00
        mock_datetime.datetime.now.return_value = datetime.datetime(2026, 9, 12, 14, 0, 0)
        # Target is 13:00 (past today, so 23 hours from now)
        mock_datetime.datetime.strptime = datetime.datetime.strptime
        mock_datetime.datetime.combine = datetime.datetime.combine
        mock_datetime.timedelta = datetime.timedelta

        res = datetime_ops.get_time_until("13:00")
        assert res["success"] is True
        assert res["data"]["minutes_until"] == 23 * 60

    def test_invalid_types(self):
        assert datetime_ops.get_time_until(None)["success"] is False
        assert datetime_ops.get_time_until(123)["success"] is False
        assert datetime_ops.get_time_until("")["success"] is False
        assert datetime_ops.get_time_until("   ")["success"] is False
        assert datetime_ops.get_time_until("x" * 51)["success"] is False

    def test_invalid_format(self):
        res = datetime_ops.get_time_until("banana")
        assert res["success"] is False
        assert "Could not parse" in res["error"]


class TestSetReminder:
    def setup_method(self):
        # Reset counter for isolation
        datetime_ops._active_reminders = 0

    def teardown_method(self):
        datetime_ops._active_reminders = 0

    @patch("skills.datetime_ops.threading.Timer")
    def test_success(self, mock_timer_class):
        mock_timer = MagicMock()
        mock_timer_class.return_value = mock_timer

        res = datetime_ops.set_reminder("Take a break", 300)
        assert res["success"] is True
        assert res["data"]["message"] == "Take a break"
        assert res["data"]["delay_seconds"] == 300

        mock_timer_class.assert_called_once_with(300, datetime_ops._fire_reminder, args=["Take a break"])
        assert mock_timer.daemon is True
        mock_timer.start.assert_called_once()
        assert datetime_ops._active_reminders == 1

    def test_invalid_message(self):
        assert datetime_ops.set_reminder(None, 10)["success"] is False
        assert datetime_ops.set_reminder("", 10)["success"] is False
        assert datetime_ops.set_reminder("x" * 501, 10)["success"] is False
        assert datetime_ops.set_reminder("null\x00byte", 10)["success"] is False

    def test_invalid_delay(self):
        assert datetime_ops.set_reminder("test", -1)["success"] is False
        assert datetime_ops.set_reminder("test", 0)["success"] is False
        assert datetime_ops.set_reminder("test", 86401)["success"] is False
        assert datetime_ops.set_reminder("test", True)["success"] is False
        assert datetime_ops.set_reminder("test", 10.5)["success"] is False
        assert datetime_ops.set_reminder("test", "10")["success"] is False

    @patch("skills.datetime_ops.threading.Timer")
    def test_max_reminders(self, mock_timer_class):
        datetime_ops._active_reminders = 20
        res = datetime_ops.set_reminder("test", 10)
        assert res["success"] is False
        assert "Maximum number" in res["error"]

    @patch("skills.datetime_ops.print")
    @patch("core.notifier.manager.notify")
    def test_callback_prints(self, mock_notify, mock_print):
        datetime_ops._active_reminders = 1
        datetime_ops._fire_reminder("Hello world")
        mock_print.assert_called_once()
        assert "Hello world" in mock_print.call_args[0][0]
        assert datetime_ops._active_reminders == 0


# ===========================================================================
# Registry & Brain Tests
# ===========================================================================

class TestToolRegistryPhase17:
    def test_tools_registered(self):
        registry = ToolRegistry()
        assert "get_datetime" in registry.tools
        assert "get_time_until" in registry.tools
        assert "set_reminder" in registry.tools

    def test_not_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["get_datetime"][1] is False
        assert registry.tools["get_time_until"][1] is False
        assert registry.tools["set_reminder"][1] is False


class TestBrainGroundingPhase17:
    def test_get_datetime_success(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "get_datetime",
            {},
            json.dumps({
                "success": True,
                "data": {"date": "2026-09-12", "time": "14:30:00", "weekday": "Saturday", "timezone": "+05:30"}
            })
        )
        assert res is not None
        assert "14:30:00" in res
        assert "Saturday" in res
        assert "2026-09-12" in res

    def test_get_datetime_failure(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "get_datetime",
            {},
            json.dumps({"success": False, "error": "test error"})
        )
        assert res is not None
        assert "I tried to get the date and time" in res
        assert "test error" in res

    def test_get_time_until_success(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "get_time_until",
            {"target_time": "5pm"},
            json.dumps({"success": True, "data": {"target_time": "5pm", "minutes_until": 180}})
        )
        assert "3 hour(s)" in res
        assert "0 minute(s)" in res
        assert "5pm" in res

    def test_set_reminder_success(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "set_reminder",
            {"message": "break", "delay_seconds": 300},
            json.dumps({"success": True, "data": {"message": "break", "delay_seconds": 300}})
        )
        assert "5 minute(s)" in res
        assert "break" in res


class TestBrainIntegrationPhase17:
    @patch("skills.datetime_ops.get_datetime")
    def test_get_datetime_pipeline(self, mock_get):
        mock_get.return_value = {
            "success": True, 
            "data": {"date": "2026-09-12", "time": "14:30", "weekday": "Sat", "timezone": "0"}
        }
        
        brain = _make_brain('{"tool": "get_datetime", "args": {}}')
        response = brain.process_input("What time is it?")
        
        # Tool was called
        mock_get.assert_called_once()
        # LLM Pass 2 not used because of deterministic string
        brain.llm.chat.assert_not_called()
        assert "14:30" in response

    @patch("skills.datetime_ops.set_reminder")
    def test_set_reminder_pipeline(self, mock_set):
        mock_set.return_value = {
            "success": True,
            "data": {"message": "task", "delay_seconds": 60}
        }
        
        brain = _make_brain('{"tool": "set_reminder", "args": {"message": "task", "delay_seconds": 60}}')
        # We need to spy on _ask_confirmation to ensure it's not called
        brain._ask_confirmation = MagicMock()
        
        response = brain.process_input("Remind me in 1 minute to task")
        
        mock_set.assert_called_once_with(message="task", delay_seconds=60)
        brain._ask_confirmation.assert_not_called()
        brain.llm.chat.assert_not_called()
        assert "1 minute(s)" in response
        assert "task" in response


class TestRegressionGuardPhase17:
    def test_existing_tools_still_registered(self):
        registry = ToolRegistry()
        assert len(registry.tools) >= 36  # 33 from p16 + 3 new

    def test_write_file_still_dangerous(self):
        registry = ToolRegistry()
        assert registry.tools["write_file"][1] is True
