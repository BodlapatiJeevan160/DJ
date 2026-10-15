import pytest
import sys
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def force_win32():
    with patch("sys.platform", "win32"):
        yield

def _make_mock_proc(poll_result):
    proc = MagicMock()
    proc.poll.return_value = poll_result
    return proc

class TestOpenApplicationRegression:
    
    def test_notepad_not_running_launches_successfully(self):
        """Notepad is not running initially. The launch succeeds and window appears."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": False}}):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = _make_mock_proc(None) # None means still running
                
                # Mock window appearance on 2nd poll
                with patch("skills.window_ops._get_all_windows", side_effect=[
                    [], # 1st check: no windows
                    [{"title": "Untitled - Notepad"}], # 2nd check: window appears
                ]):
                    with patch("time.sleep"):
                        res = launch("notepad")
                        assert res["success"] is True
                        assert res["data"]["launched"] == "notepad.exe"
                        assert "error" in res and res["error"] is None

    def test_notepad_not_running_fails_to_open_window(self):
        """Notepad is not running, process starts but no window appears within timeout."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": False}}):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = _make_mock_proc(None) # None means still running
                
                # Mock window never appears
                with patch("skills.window_ops._get_all_windows", return_value=[]):
                    with patch("time.sleep"):
                        res = launch("notepad")
                        assert res["success"] is False
                        assert "Process started but no visible window could be verified" in res["error"]

    def test_notepad_running_with_visible_window(self):
        """Notepad is already running, and a visible window exists. It should focus the window."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": True}}):
            with patch("skills.window_ops._get_all_windows", return_value=[{"title": "Untitled - Notepad", "hwnd": 123}]):
                with patch("skills.window_ops.focus_window", return_value={"success": True}) as mock_focus:
                    res = launch("notepad")
                    assert res["success"] is True
                    assert res["data"]["already_running"] is True
                    assert res["data"]["focused"] is True
                    mock_focus.assert_called_once_with("Notepad")

    def test_notepad_running_without_visible_window(self):
        """Notepad is already running, but NO visible window exists (e.g. background process). It should launch a new one."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": True}}):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = _make_mock_proc(None)
                
                # First get_all_windows called for running check -> no notepad windows
                # Subsequent called during launch loop -> window appears
                with patch("skills.window_ops._get_all_windows", side_effect=[
                    [], # Check if visible window exists -> no
                    [{"title": "Untitled - Notepad"}], # Launch verification -> yes
                ]):
                    with patch("time.sleep"):
                        res = launch("notepad")
                        assert res["success"] is True
                        assert "launched" in res["data"]
                        mock_popen.assert_called_once()

    def test_focus_failure_returns_failure(self):
        """Notepad is running and visible, but focusing fails (e.g. ambiguous match)."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": True}}):
            with patch("skills.window_ops._get_all_windows", return_value=[{"title": "Untitled - Notepad", "hwnd": 123}]):
                with patch("skills.window_ops.focus_window", return_value={"success": False, "error": "Ambiguous"}):
                    res = launch("notepad")
                    assert res["success"] is False
                    assert "Window exists but could not be focused" in res["error"]

    def test_window_enumeration_failure(self):
        """Exception raised during initial window enumeration when app is running."""
        from skills.app_registry import launch
        with patch("skills.app_registry.get_status", return_value={"success": True, "data": {"running": True}}):
            with patch("skills.window_ops._get_all_windows", side_effect=Exception("API failure")):
                res = launch("notepad")
                assert res["success"] is False
                assert "Failed during window enumeration: API failure" in res["error"]
