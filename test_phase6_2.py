"""
test_phase6_2.py
================
Phase 6.2 — Secure Application Closing Tests.

All unit tests mock psutil so no real processes are terminated.
Validates the allowlist logic, target process mapping, and failure states.

Run with:
    python test_phase6_2.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch

def make_brain(intent_json: str, llm_chat_response: str = "N/A"):
    from core.brain import Brain
    brain = Brain()
    brain.llm.chat_json = MagicMock(return_value=intent_json)
    brain.llm.chat      = MagicMock(return_value=llm_chat_response)
    brain._ask_confirmation = MagicMock(return_value=True)
    return brain


class MockProcess:
    def __init__(self, name):
        self.info = {'name': name}
        self.terminate = MagicMock()


class TestPhase6_2SafeClose(unittest.TestCase):

    # -----------------------------------------------------------------------
    # TEST 1: Close an allowed application (mocked success)
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_close_chrome_success(self, mock_process_iter):
        p1 = MockProcess("chrome.exe")
        p2 = MockProcess("chrome.exe")
        p3 = MockProcess("explorer.exe")
        mock_process_iter.return_value = [p1, p2, p3]

        brain = make_brain('{"tool": "close_application", "args": {"application": "chrome"}}')
        
        # We don't mock execute_tool here; we want to test the full pipeline through app_registry
        response = brain.process_input("Terminate Chrome")

        p1.terminate.assert_called_once()
        p2.terminate.assert_called_once()
        p3.terminate.assert_not_called()

        self.assertIn("Done! I've closed Google Chrome", response)
        print(f"\n[TEST 1] Chrome closed successfully: {response}")

    # -----------------------------------------------------------------------
    # TEST 2: Close an application that is not running
    # -----------------------------------------------------------------------
    @patch('psutil.process_iter')
    def test_close_notepad_not_running(self, mock_process_iter):
        p1 = MockProcess("explorer.exe")
        mock_process_iter.return_value = [p1]

        brain = make_brain('{"tool": "close_application", "args": {"application": "notepad"}}')
        response = brain.process_input("Terminate Notepad")

        self.assertIn("failed", response.lower())
        self.assertIn("Notepad is not currently running", response)
        print(f"\n[TEST 2] App not running handled gracefully: {response}")

    # -----------------------------------------------------------------------
    # TEST 3: Unknown identifier rejected
    # -----------------------------------------------------------------------
    def test_unknown_app_rejected(self):
        from skills.app_registry import terminate
        result = terminate("hackertool")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 3] Unknown app rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 4: Path traversal rejected
    # -----------------------------------------------------------------------
    def test_path_traversal_rejected(self):
        from skills.app_registry import terminate
        result = terminate("../../Windows/system32/cmd")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 4] Path traversal rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 5: Shell injection rejected
    # -----------------------------------------------------------------------
    def test_shell_injection_rejected(self):
        from skills.app_registry import terminate
        result = terminate("notepad; del C:\\")
        self.assertFalse(result["success"])
        self.assertIn("permission to manage", result["error"])
        print(f"\n[TEST 5] Shell injection rejected: {result['error'][:80]}...")

    # -----------------------------------------------------------------------
    # TEST 6: Existing open_application still works
    # -----------------------------------------------------------------------
    @patch('time.sleep')
    @patch('skills.window_ops._get_all_windows')
    @patch('skills.app_registry.get_status')
    @patch('subprocess.Popen')
    def test_open_application_regression(self, mock_popen, mock_get_status, mock_get_all_windows, mock_sleep):
        mock_get_status.return_value = {"success": True, "data": {"running": False}}
        mock_popen.return_value.poll.return_value = None
        mock_get_all_windows.return_value = [{"hwnd": 1, "title": "Untitled - Notepad", "pid": 1234}]
        brain = make_brain('{"tool": "open_application", "args": {"application": "notepad"}}')
        response = brain.process_input("Open Notepad")
        mock_popen.assert_any_call(["notepad.exe"], shell=False)
        self.assertIn("Done! I've opened Notepad", response)
        print(f"\n[TEST 6] open_application regression OK.")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6.2 — Secure Application Closing Tests")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromTestCase(TestPhase6_2SafeClose)
    )
    sys.exit(0 if result.wasSuccessful() else 1)
