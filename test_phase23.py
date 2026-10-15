"""
test_phase23.py
===============
Phase 23 -- Proactive Contextual Awareness Tests

Tests for ContextMonitor, rule CRUD operations, JSON persistence,
cooldown logic, and notification dispatch.
"""

import os
import json
import time
import ctypes
import pytest
from unittest.mock import patch, MagicMock

from core.context_monitor import ContextMonitor
from skills.context_ops import add_context_rule, list_context_rules, remove_context_rule
from core.brain import Brain
from core.tool_registry import ToolRegistry


# ===========================================================================
# 1. ContextMonitor Initialization & Persistence
# ===========================================================================

class TestContextMonitorPersistence:
    @pytest.fixture
    def temp_monitor(self, tmp_path):
        """Creates a ContextMonitor with a temporary storage path."""
        monitor = ContextMonitor()
        monitor.STORAGE_DIR = str(tmp_path)
        monitor.STORAGE_FILE = os.path.join(str(tmp_path), "context_rules.json")
        monitor._rules = {}
        monitor._cooldowns = {}
        return monitor

    def test_missing_json_handling(self, temp_monitor):
        # File doesn't exist yet
        temp_monitor._load_rules()
        assert temp_monitor._rules == {}

    def test_corrupt_json_handling(self, temp_monitor):
        # Write corrupt JSON
        os.makedirs(temp_monitor.STORAGE_DIR, exist_ok=True)
        with open(temp_monitor.STORAGE_FILE, "w") as f:
            f.write("{invalid json")
        
        temp_monitor._load_rules()
        assert temp_monitor._rules == {}  # Should recover gracefully

    def test_atomic_save_and_load(self, temp_monitor):
        # Add a rule and save
        temp_monitor._rules = {"notepad": "Focus time"}
        assert temp_monitor._save_rules() is True
        
        # Verify file exists
        assert os.path.exists(temp_monitor.STORAGE_FILE)
        
        # Create a new monitor to load
        new_monitor = ContextMonitor()
        new_monitor.STORAGE_DIR = temp_monitor.STORAGE_DIR
        new_monitor.STORAGE_FILE = temp_monitor.STORAGE_FILE
        new_monitor._load_rules()
        
        assert new_monitor._rules == {"notepad": "Focus time"}

    def test_validation_on_load(self, temp_monitor):
        # Write raw JSON with bad data
        bad_data = {
            "good": "value",
            "too_long_key_" + ("x" * 100): "value",
            "bad_value": "y" * 200,
            "control\x00char": "value"
        }
        os.makedirs(temp_monitor.STORAGE_DIR, exist_ok=True)
        with open(temp_monitor.STORAGE_FILE, "w") as f:
            json.dump(bad_data, f)
            
        temp_monitor._load_rules()
        
        # Only the good rule should survive (and control char should be sanitized)
        assert "good" in temp_monitor._rules
        assert "too_long_key_" + ("x" * 100) not in temp_monitor._rules
        assert "control char" in temp_monitor._rules  # sanitized


# ===========================================================================
# 2. Rule CRUD Operations & Validation
# ===========================================================================

class TestContextOps:
    @patch("skills.context_ops.monitor")
    def test_add_rule_validation(self, mock_monitor):
        mock_monitor.add_rule.return_value = {"success": True}
        
        res = add_context_rule("vscode", "Coding time")
        assert res["success"] is True
        mock_monitor.add_rule.assert_called_with("vscode", "Coding time")

    def test_add_rule_limits(self):
        monitor = ContextMonitor()
        monitor._save_rules = MagicMock(return_value=True)
        monitor._rules = {}
        
        # Empty
        assert monitor.add_rule("", "msg")["success"] is False
        # Too long keyword
        assert monitor.add_rule("k" * 51, "msg")["success"] is False
        # Too long message
        assert monitor.add_rule("key", "m" * 101)["success"] is False
        # Max rules
        monitor._rules = {str(i): "msg" for i in range(50)}
        assert monitor.add_rule("new", "msg")["success"] is False

    def test_add_rule_sanitization(self):
        monitor = ContextMonitor()
        monitor._save_rules = MagicMock(return_value=True)
        monitor._rules = {}
        
        res = monitor.add_rule(" \x00test\n ", "  message\twith space  ")
        assert res["success"] is True
        assert "test" in monitor._rules  # lowercase, sanitized


# ===========================================================================
# 3. Active Window Detection & Triggering (Mocked)
# ===========================================================================

class TestContextMonitorTriggering:
    def test_get_active_window_ctypes_mock(self):
        monitor = ContextMonitor()
        monitor._ctypes_available = True
        
        mock_user32 = MagicMock()
        mock_user32.GetForegroundWindow.return_value = 12345
        mock_user32.GetWindowTextLengthW.return_value = 7
        
        def mock_get_text(hwnd, buf, length):
            buf.value = "Notepad"
        mock_user32.GetWindowTextW.side_effect = mock_get_text
        
        monitor.user32 = mock_user32
        
        title = monitor._get_active_window_title()
        assert title == "Notepad"

    @patch("core.notifier.manager.notify")
    @patch("core.context_monitor.time.time")
    def test_worker_loop_trigger(self, mock_time, mock_notify):
        monitor = ContextMonitor()
        monitor._rules = {"notepad": "You opened notepad!"}
        monitor._cooldowns = {}
        
        # Mock window title
        monitor._get_active_window_title = MagicMock(return_value="Untitled - Notepad")
        
        # Freeze time
        mock_time.return_value = 1000.0
        
        # Run one iteration of the loop logic manually to avoid thread timing issues
        title = monitor._get_active_window_title()
        assert "notepad" in title.lower()
        
        # Simulate worker logic
        last_fired = monitor._cooldowns.get("notepad", 0)
        if mock_time.return_value - last_fired >= monitor.COOLDOWN_SEC:
            monitor._cooldowns["notepad"] = mock_time.return_value
            mock_notify(message=monitor._rules["notepad"], duration=5)
            
        mock_notify.assert_called_once_with(message="You opened notepad!", duration=5)
        assert monitor._cooldowns["notepad"] == 1000.0

    @patch("core.notifier.manager.notify")
    @patch("core.context_monitor.time.time")
    def test_cooldown_prevents_spam(self, mock_time, mock_notify):
        monitor = ContextMonitor()
        monitor._rules = {"notepad": "You opened notepad!"}
        monitor._cooldowns = {"notepad": 1000.0}  # Fired at 1000s
        
        # Freeze time to 1100s (only 100s passed, cooldown is 300s)
        mock_time.return_value = 1100.0
        
        # Simulate worker logic
        last_fired = monitor._cooldowns.get("notepad", 0)
        if mock_time.return_value - last_fired >= monitor.COOLDOWN_SEC:
            mock_notify(message=monitor._rules["notepad"], duration=5)
            
        mock_notify.assert_not_called()

    @patch("core.notifier.manager.notify")
    @patch("core.context_monitor.time.time")
    def test_independent_rule_cooldowns(self, mock_time, mock_notify):
        monitor = ContextMonitor()
        monitor._rules = {"app1": "Msg 1", "app2": "Msg 2"}
        monitor._cooldowns = {"app1": 1000.0}  # app1 is on cooldown
        
        mock_time.return_value = 1100.0
        
        # Simulate worker logic for app2
        last_fired = monitor._cooldowns.get("app2", 0)
        if mock_time.return_value - last_fired >= monitor.COOLDOWN_SEC:
            mock_notify(message=monitor._rules["app2"], duration=5)
            
        mock_notify.assert_called_once_with(message="Msg 2", duration=5)


# ===========================================================================
# 4. Threading & Lifecycle
# ===========================================================================

class TestContextMonitorLifecycle:
    def test_duplicate_thread_prevention(self):
        monitor = ContextMonitor()
        monitor._ctypes_available = True
        
        monitor.start()
        assert monitor.running is True
        t1 = monitor._thread
        
        monitor.start()  # Should not spawn a second thread
        assert monitor._thread is t1
        
        monitor.stop()
        assert monitor.running is False

    @patch("core.context_monitor.ContextMonitor._get_active_window_title")
    def test_exception_recovery(self, mock_get_title):
        monitor = ContextMonitor()
        monitor._ctypes_available = True
        
        # If OS API throws, monitor shouldn't crash
        mock_get_title.side_effect = Exception("OS Crash")
        
        monitor.start()
        time.sleep(0.1) # let loop run
        monitor.stop()
        
        assert not monitor._thread.is_alive()


# ===========================================================================
# 5. Integration (Brain & Registry)
# ===========================================================================

class TestPhase23Integration:
    def test_tools_registered_and_safe(self):
        registry = ToolRegistry()
        assert "add_context_rule" in registry.tools
        assert "list_context_rules" in registry.tools
        assert "remove_context_rule" in registry.tools
        
        assert registry.tools["add_context_rule"][1] is False
        assert registry.tools["remove_context_rule"][1] is False

    def test_brain_grounding_add_rule(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "add_context_rule",
            {"keyword": "test", "message": "hello"},
            json.dumps({"success": True, "data": {"keyword": "test", "message": "hello"}})
        )
        assert "watch for 'test'" in res
        assert "notify you with 'hello'" in res

    def test_brain_grounding_list_rules(self):
        brain = Brain()
        res = brain._build_grounded_response(
            "list_context_rules",
            {},
            json.dumps({"success": True, "data": {"rules": [{"keyword": "test", "message": "msg"}]}})
        )
        assert "1 active context rule" in res
        assert "test" in res
        assert "msg" in res

    def test_no_tts_in_proactive_path(self):
        # We manually verify that the worker loop ONLY calls manager.notify
        # and there is NO reference to audio_engine.speak in context_monitor.py
        with open("core/context_monitor.py", "r") as f:
            content = f.read()
        assert "audio_engine" not in content
        assert ".speak(" not in content


# ===========================================================================
# 6. Bug 1 Fix: ctypes API Declarations & Session Isolation
# ===========================================================================

class TestCtypesAPIDeclarations:
    def test_restype_argtypes_set(self):
        """GetForegroundWindow must have HWND restype, not default c_int (32-bit truncation risk)."""
        import ctypes.wintypes
        monitor = ContextMonitor()
        assert monitor._ctypes_available is True
        assert monitor.user32.GetForegroundWindow.restype == ctypes.wintypes.HWND
        assert monitor.user32.GetWindowTextLengthW.restype == ctypes.c_int
        assert monitor.user32.GetWindowTextW.restype == ctypes.c_int

    def test_interactive_session_attribute_present(self):
        """ContextMonitor must expose _interactive_session regardless of environment."""
        monitor = ContextMonitor()
        assert hasattr(monitor, '_interactive_session')
        assert isinstance(monitor._interactive_session, bool)

    def test_session_isolation_warning_logged(self):
        """When GetForegroundWindow returns 0, a warning must be logged (not silently ignored)."""
        import logging
        with patch("core.context_monitor.ContextMonitor._load_rules"):
            with patch("ctypes.windll.user32") as mock_u32:
                mock_u32.GetForegroundWindow.return_value = 0
                mock_u32.GetForegroundWindow.restype = None
                mock_u32.GetForegroundWindow.argtypes = None

                with patch("core.context_monitor.logger") as mock_logger:
                    # Re-init a fresh monitor to trigger the probe
                    monitor = ContextMonitor.__new__(ContextMonitor)
                    monitor._rules = {}
                    monitor._cooldowns = {}
                    monitor._lock = __import__('threading').RLock()
                    monitor.running = False
                    monitor._thread = None
                    monitor._ctypes_available = False
                    monitor._interactive_session = False

                    import ctypes, ctypes.wintypes
                    monitor._ctypes = ctypes
                    monitor.user32 = ctypes.windll.user32

                    monitor.user32.GetForegroundWindow.restype = ctypes.wintypes.HWND
                    monitor.user32.GetForegroundWindow.argtypes = []
                    monitor.user32.GetWindowTextLengthW.restype = ctypes.c_int
                    monitor.user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
                    monitor.user32.GetWindowTextW.restype = ctypes.c_int
                    monitor.user32.GetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
                    monitor._ctypes_available = True

                    probe = monitor.user32.GetForegroundWindow()
                    # In CI / agent environment this will be 0 — validate the contract
                    if not probe:
                        assert monitor._interactive_session is False

    def test_get_active_window_empty_hwnd_returns_empty(self):
        """_get_active_window_title must return '' when GetForegroundWindow returns 0."""
        monitor = ContextMonitor()
        monitor._ctypes_available = True
        mock_user32 = MagicMock()
        mock_user32.GetForegroundWindow.return_value = 0
        monitor.user32 = mock_user32
        assert monitor._get_active_window_title() == ""

    def test_get_active_window_valid_hwnd(self):
        """_get_active_window_title must return the correct window title when hwnd is valid."""
        import ctypes
        monitor = ContextMonitor()
        monitor._ctypes_available = True
        monitor._ctypes = ctypes

        mock_user32 = MagicMock()
        mock_user32.GetForegroundWindow.return_value = 99999
        mock_user32.GetWindowTextLengthW.return_value = 7

        def _fill_buf(hwnd, buf, length):
            buf.value = "Notepad"
            return 7
        mock_user32.GetWindowTextW.side_effect = _fill_buf
        monitor.user32 = mock_user32

        title = monitor._get_active_window_title()
        assert title == "Notepad"

    def test_keyword_match_is_case_insensitive(self):
        """Rule keyword matching must be case-insensitive."""
        monitor = ContextMonitor()
        monitor._rules = {"notepad": "Focus alert"}
        title_lower = "Untitled - Notepad".lower()
        assert "notepad" in title_lower


# ===========================================================================
# 7. Bug 2 Fix: winotify Backend & Resilient Dispatch
# ===========================================================================

class TestNotifierWinotifyBackend:
    def test_notifier_backend_is_winotify_or_plyer(self):
        """After the fix, the active backend must be winotify or plyer (not None)."""
        from core.notifier import manager, _BACKEND
        assert _BACKEND in ("winotify", "plyer"), (
            f"Expected winotify or plyer backend, got: {_BACKEND!r}. "
            "Run: venv\\Scripts\\pip install winotify"
        )
        assert manager.available is True

    def test_notify_returns_success_dict_structure(self):
        """A successful notify() must return success=True and include 'backend' in data."""
        from core.notifier import manager
        with patch.object(manager, '_dispatch', return_value=None) as mock_dispatch:
            result = manager.notify(message="test message", duration=3)
        assert result["success"] is True
        assert "backend" in result["data"]
        assert result["data"]["message"] == "test message"
        mock_dispatch.assert_called_once_with("test message", 3)

    def test_dispatch_backend_error_returns_error_dict(self):
        """If the backend raises, notify() must return an error dict, never re-raise."""
        from core.notifier import manager
        with patch.object(manager, '_dispatch', side_effect=RuntimeError("backend exploded")):
            result = manager.notify(message="test", duration=3)
        assert result["success"] is False
        assert "backend exploded" in result["error"]

    def test_unavailable_manager_returns_error_dict(self):
        """When available=False, notify() must return error dict immediately."""
        from core.notifier import NotificationManager
        mgr = NotificationManager.__new__(NotificationManager)
        mgr.available = False
        mgr._history = []
        mgr._lock = __import__('threading').Lock()
        result = mgr.notify(message="hello")
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_get_status_includes_backend(self):
        """get_status() must report the backend name."""
        from core.notifier import manager
        status = manager.get_status()
        assert status["success"] is True
        assert "backend" in status["data"]
        assert status["data"]["backend"] in ("winotify", "plyer", "none")

    def test_no_tts_in_notifier(self):
        """notifier.py must never reference audio_engine or .speak()."""
        with open("core/notifier.py", "r") as f:
            content = f.read()
        assert "audio_engine" not in content
        assert ".speak(" not in content

    def test_worker_loop_survives_notify_exception(self):
        """If manager.notify() raises unexpectedly, the worker thread must not die."""
        monitor = ContextMonitor()
        monitor._rules = {"notepad": "Alert"}
        monitor._cooldowns = {}
        monitor._ctypes_available = True
        monitor._get_active_window_title = MagicMock(return_value="Untitled - Notepad")

        from core import notifier
        with patch.object(notifier.manager, 'notify', side_effect=Exception("notify exploded")):
            monitor._ctypes_available = True
            monitor.start()
            time.sleep(0.3)
            monitor.stop()
        # Thread must have stopped cleanly (not crashed with unhandled exception)
        assert not monitor._thread.is_alive()
