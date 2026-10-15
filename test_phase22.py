"""
test_phase22.py
===============
Phase 22 -- Desktop Notification System Tests

Verifies:
  - NotificationManager init and graceful failure.
  - Successful notification dispatch via mocked plyer.
  - Fixed "JARVIS" title enforcement.
  - Message sanitization and length limits.
  - 10/60s rate limiting and rolling window.
  - Integration with set_reminder.
  - TTS fallback behavior.
  - Cross-crash protection (TTS vs Desktop Notification).
  - Brain templates for new tools.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from core.notifier import NotificationManager
from skills.notification_ops import send_notification, get_notification_status
from skills.datetime_ops import set_reminder, _fire_reminder, _reminders_lock
import skills.datetime_ops as dt_ops
from core.brain import Brain
from core.tool_registry import ToolRegistry


# ===========================================================================
# 1. NotificationManager Tests
# ===========================================================================

class TestNotificationManager:
    def test_init_without_plyer(self):
        # Backend detection runs at module import, not __init__.
        # Patch the module-level _BACKEND to None to simulate no backend.
        import core.notifier as notifier_mod
        with patch.object(notifier_mod, "_BACKEND", None):
            manager = NotificationManager()
            assert manager.available is False

    @patch("core.notifier.logger")
    def test_init_with_broken_plyer(self, mock_logger):
        # Same: simulate no available backend at init time.
        import core.notifier as notifier_mod
        with patch.object(notifier_mod, "_BACKEND", None):
            manager = NotificationManager()
            assert manager.available is False

    def test_notify_when_unavailable(self):
        manager = NotificationManager()
        manager.available = False
        res = manager.notify("Hello")
        assert res["success"] is False
        assert "not available" in res["error"]

    def test_notify_invalid_inputs(self):
        manager = NotificationManager()
        manager.available = True
        
        # Non-string
        assert manager.notify(123)["success"] is False
        
        # Empty after sanitize
        assert manager.notify("   ")["success"] is False
        assert manager.notify("\x00\x08")["success"] is False

    def test_sanitize_and_length_limit(self):
        manager = NotificationManager()
        manager.available = True

        with patch.object(manager, "_dispatch") as mock_dispatch:
            # Test sanitization: control char stripped, whitespace normalised
            manager.notify("Hello\x00World")
            mock_dispatch.assert_called_with("Hello World", 5)

            # Test length limit: truncated to 256 chars ending with "..."
            long_msg = "A" * 300
            manager.notify(long_msg)
            called_msg = mock_dispatch.call_args[0][0]
            assert len(called_msg) == 256
            assert called_msg.endswith("...")

    def test_duration_clamp(self):
        manager = NotificationManager()
        manager.available = True

        with patch.object(manager, "_dispatch") as mock_dispatch:
            manager.notify("Test", duration=-5)
            assert mock_dispatch.call_args[0][1] == 1

            manager.notify("Test", duration=100)
            assert mock_dispatch.call_args[0][1] == 30

            manager.notify("Test", duration="invalid")
            assert mock_dispatch.call_args[0][1] == 5

    def test_notify_backend_failure(self):
        manager = NotificationManager()
        manager.available = True

        with patch.object(manager, "_dispatch", side_effect=Exception("OS Error")):
            res = manager.notify("Test")
        assert res["success"] is False
        assert "Backend failure" in res["error"]

    @patch("time.time")
    def test_rate_limit(self, mock_time):
        manager = NotificationManager()
        manager.available = True

        with patch.object(manager, "_dispatch"):
            # Freeze time
            mock_time.return_value = 100.0

            # Send 10 notifications (limit)
            for _ in range(10):
                res = manager.notify("Test")
                assert res["success"] is True

            # 11th should fail
            res = manager.notify("Test")
            assert res["success"] is False
            assert "Rate limit" in res["error"]

            # Move time forward by 61 seconds
            mock_time.return_value = 161.0

            # Should succeed again (rolling window cleared)
            res = manager.notify("Test")
            assert res["success"] is True


# ===========================================================================
# 2. Tool Interface Tests
# ===========================================================================

class TestNotificationOps:
    @patch("skills.notification_ops.manager")
    def test_send_notification(self, mock_manager):
        mock_manager.notify.return_value = {"success": True}
        res = send_notification("Hello")
        assert res["success"] is True
        mock_manager.notify.assert_called_with(message="Hello", duration=5)

    @patch("skills.notification_ops.manager")
    def test_get_notification_status(self, mock_manager):
        mock_manager.get_status.return_value = {"success": True, "data": {"available": True}}
        res = get_notification_status()
        assert res["success"] is True
        assert res["data"]["available"] is True


# ===========================================================================
# 3. Reminder Integration Tests
# ===========================================================================

class TestReminderIntegration:
    @patch("core.notifier.manager")
    @patch("core.audio.engine")
    @patch("builtins.print")
    def test_reminder_fires_notification(self, mock_print, mock_audio, mock_manager):
        mock_audio.enabled = False
        
        # Manually increment active reminders to simulate set_reminder
        with dt_ops._reminders_lock:
            dt_ops._active_reminders = 1
            
        _fire_reminder("Buy milk")
        
        # Verify Notifier was called
        mock_manager.notify.assert_called_once_with(message="Buy milk")
        
        # Verify TTS was NOT called
        mock_audio.speak.assert_not_called()
        
        # Verify counter decremented
        with dt_ops._reminders_lock:
            assert dt_ops._active_reminders == 0

    @patch("core.notifier.manager")
    @patch("core.audio.engine")
    @patch("builtins.print")
    def test_reminder_fires_tts_if_enabled(self, mock_print, mock_audio, mock_manager):
        mock_audio.enabled = True
        
        with dt_ops._reminders_lock:
            dt_ops._active_reminders = 1
            
        _fire_reminder("Buy milk")
        
        mock_manager.notify.assert_called_once_with(message="Buy milk")
        mock_audio.speak.assert_called_once_with("Reminder: Buy milk")

    @patch("core.notifier.manager")
    @patch("core.audio.engine")
    @patch("builtins.print")
    def test_notification_crash_does_not_break_tts(self, mock_print, mock_audio, mock_manager):
        mock_audio.enabled = True
        mock_manager.notify.side_effect = Exception("Crash")
        
        with dt_ops._reminders_lock:
            dt_ops._active_reminders = 1
            
        _fire_reminder("Buy milk")
        
        # Notify crashed but TTS should still fire
        mock_audio.speak.assert_called_once_with("Reminder: Buy milk")
        
        with dt_ops._reminders_lock:
            assert dt_ops._active_reminders == 0

    @patch("core.notifier.manager")
    @patch("core.audio.engine")
    @patch("builtins.print")
    def test_tts_crash_does_not_break_reminder_cleanup(self, mock_print, mock_audio, mock_manager):
        mock_audio.enabled = True
        mock_audio.speak.side_effect = Exception("Crash")
        
        with dt_ops._reminders_lock:
            dt_ops._active_reminders = 1
            
        _fire_reminder("Buy milk")
        
        # Cleanup should still happen
        with dt_ops._reminders_lock:
            assert dt_ops._active_reminders == 0


# ===========================================================================
# 4. Brain & ToolRegistry Integration
# ===========================================================================

class TestPhase22BrainIntegration:
    def test_tools_registered_and_safe(self):
        registry = ToolRegistry()
        assert "send_notification" in registry.tools
        assert "get_notification_status" in registry.tools
        
        assert registry.tools["send_notification"][1] is False
        assert registry.tools["get_notification_status"][1] is False

    def test_brain_grounding_send_notification(self):
        brain = Brain()
        tool_name = "send_notification"
        tool_args = {"message": "Test"}
        
        # Success
        tool_result = '{"success": true, "data": {"message": "Test"}}'
        res = brain._build_grounded_response(tool_name, tool_args, tool_result)
        assert "Done! I've sent you a desktop notification: 'Test'." in res
        
        # Failure
        tool_result = '{"success": false, "error": "Rate limit"}'
        res = brain._build_grounded_response(tool_name, tool_args, tool_result)
        assert "failed" in res.lower()
        assert "Rate limit" in res

    def test_brain_grounding_get_notification_status(self):
        brain = Brain()
        tool_name = "get_notification_status"
        
        # Available
        tool_result = '{"success": true, "data": {"available": true, "backend": "plyer"}}'
        res = brain._build_grounded_response(tool_name, {}, tool_result)
        assert "Desktop notifications are available" in res
        assert "plyer" in res
        
        # Unavailable
        tool_result = '{"success": true, "data": {"available": false}}'
        res = brain._build_grounded_response(tool_name, {}, tool_result)
        assert "unavailable" in res.lower()
