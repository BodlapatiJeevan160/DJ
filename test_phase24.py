"""
test_phase24.py
===============
Phase 24 -- Window Management Tests
"""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_user32(windows: list[dict] = None):
    windows = windows or []
    u32 = MagicMock()
    u32.IsWindowVisible.return_value = True
    u32.GetWindowTextLengthW.side_effect = lambda hwnd: len(
        windows[hwnd]["title"] if hwnd < len(windows) else ""
    )

    def fake_get_text(hwnd, buf, length):
        title = windows[hwnd]["title"] if hwnd < len(windows) else ""
        import ctypes
        ctypes.memmove(buf, (title + "\x00").encode("utf-16-le"),
                       (len(title) + 1) * 2)

    u32.GetWindowTextW.side_effect = fake_get_text

    def fake_enum(proc, lparam):
        for i in range(len(windows)):
            proc(i, lparam)
        return True

    u32.EnumWindows.side_effect = fake_enum
    u32.GetWindowThreadProcessId.return_value = 0
    u32.GetSystemMetrics.side_effect = lambda idx: 1920 if idx == 0 else 1080
    return u32


class TestWindowOpsPlatformGuard:
    @pytest.fixture(autouse=True)
    def force_non_windows(self):
        with patch("skills.window_ops.sys") as mock_sys:
            mock_sys.platform = "linux"
            yield

    def test_list_windows_non_windows(self):
        from skills.window_ops import list_windows
        result = list_windows()
        assert result["success"] is False
        assert "Windows" in result["error"]
        assert result["tool"] == "list_windows"

    def test_focus_window_non_windows(self):
        from skills.window_ops import focus_window
        result = focus_window("Notepad")
        assert result["success"] is False
        assert result["tool"] == "focus_window"


class TestSnapPositionValidation:
    @pytest.fixture(autouse=True)
    def patch_platform(self):
        with patch("skills.window_ops.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=[]):
                yield

    def test_invalid_position_center(self):
        from skills.window_ops import snap_window
        r = snap_window("Notepad", "center")
        assert r["success"] is False
        assert "Invalid snap position" in r["error"]
        assert "center" in r["error"]


class TestWindowResolver:
    def test_exact_match(self):
        from skills.window_ops import focus_window
        windows = [{"hwnd": 1, "title": "Notepad", "pid": 100}]
        with patch("skills.window_ops.sys") as ms:
            ms.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=windows):
                with patch("skills.window_ops._user32"):
                    r = focus_window("Notepad")
                    assert r["success"] is True

    def test_ambiguous_match_rejected(self):
        from skills.window_ops import minimize_window
        windows = [
            {"hwnd": 1, "title": "Notepad", "pid": 100},
            {"hwnd": 2, "title": "Notepad - test.txt", "pid": 101},
        ]
        with patch("skills.window_ops.sys") as ms:
            ms.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=windows):
                r = minimize_window("Notepad")
                assert r["success"] is True

    def test_genuinely_ambiguous_match_rejected(self):
        from skills.window_ops import minimize_window
        windows = [
            {"hwnd": 1, "title": "Untitled - Notepad", "pid": 100},
            {"hwnd": 2, "title": "Untitled - Notepad", "pid": 101},
        ]
        with patch("skills.window_ops.sys") as ms:
            ms.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=windows):
                r = minimize_window("Untitled - Notepad")
                assert r["success"] is False
                assert "Ambiguous" in r["error"]


class TestCloseWindowGraceful:
    def test_close_window_uses_wm_close(self):
        from skills.window_ops import close_window, WM_CLOSE
        windows = [{"hwnd": 99, "title": "Notepad", "pid": 100}]
        with patch("skills.window_ops.sys") as ms:
            ms.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=windows):
                mock_u32 = MagicMock()
                with patch("skills.window_ops._user32", mock_u32):
                    r = close_window("Notepad")
                    assert r["success"] is True
                    mock_u32.PostMessageW.assert_called_once_with(99, WM_CLOSE, 0, 0)
                    assert not hasattr(mock_u32, "TerminateProcess") or not mock_u32.TerminateProcess.called

class TestSnapWindow:
    def test_snap_left(self):
        from skills.window_ops import snap_window
        windows = [{"hwnd": 5, "title": "Notepad", "pid": 100}]
        with patch("skills.window_ops.sys") as ms:
            ms.platform = "win32"
            with patch("skills.window_ops._get_all_windows", return_value=windows):
                mock_u32 = MagicMock()
                mock_u32.GetSystemMetrics.side_effect = lambda idx: 1920 if idx == 0 else 1080
                with patch("skills.window_ops._user32", mock_u32):
                    r = snap_window("Notepad", "left")
                    assert r["success"] is True
                    mock_u32.SetWindowPos.assert_called_once_with(5, 0, 0, 0, 960, 1080, 0x0040)


class TestToolRegistration:
    def test_all_window_tools_registered(self):
        from core.tool_registry import ToolRegistry
        r = ToolRegistry()
        expected = {
            "list_windows", "focus_window", "minimize_window",
            "maximize_window", "restore_window", "close_window", "snap_window"
        }
        assert expected.issubset(set(r.tools.keys()))

    def test_close_window_is_dangerous(self):
        from core.tool_registry import ToolRegistry
        r = ToolRegistry()
        _, is_dangerous = r.tools["close_window"]
        assert is_dangerous is True


class TestBrainWindowFastPaths:
    @pytest.fixture(autouse=True)
    def no_llm(self):
        with patch("core.brain.LLMClient") as MockLLM:
            MockLLM.return_value.chat.side_effect = AssertionError("LLM called")
            yield

    def test_list_windows_fast_path(self):
        from core.brain import Brain
        brain = Brain()
        brain.history = []
        with patch.object(brain.registry, "execute_tool", return_value=json.dumps({"success": True, "data": {"windows": [], "count": 0}})) as mock_exec:
            brain.process_input("list windows")
            assert mock_exec.call_args[1]["name"] == "list_windows"

    def test_snap_left_fast_path(self):
        from core.brain import Brain
        brain = Brain()
        brain.history = []
        with patch.object(brain.registry, "execute_tool", return_value=json.dumps({"success": True, "data": {"app_name": "Notepad", "position": "left"}})) as mock_exec:
            brain.process_input("snap Notepad to the left")
            assert mock_exec.call_args[1]["name"] == "snap_window"
            assert mock_exec.call_args[1]["args"]["position"] == "left"


class TestSecurityBoundaries:
    def test_close_window_confirmation_required(self):
        from core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        deny = lambda name, args: False
        result = registry.execute_tool("close_window", {"app_name": "Notepad"}, confirm_callback=deny)
        assert "denied" in result.lower() or "User denied" in result
